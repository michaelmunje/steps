import struct
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader

CAMERAS = 6
BAYER = {
    "bayer_rggb8": cv2.COLOR_BayerBG2BGR, "bayer_rg8": cv2.COLOR_BayerBG2BGR,
    "bayer_bggr8": cv2.COLOR_BayerRG2BGR, "bayer_bg8": cv2.COLOR_BayerRG2BGR,
    "bayer_gbrg8": cv2.COLOR_BayerGR2BGR, "bayer_gb8": cv2.COLOR_BayerGR2BGR,
    "bayer_grbg8": cv2.COLOR_BayerGB2BGR, "bayer_gr8": cv2.COLOR_BayerGB2BGR,
}


class Cdr:
    def __init__(self, data):
        self.data = memoryview(data)
        self.endian = "<" if bytes(self.data[:2]) == b"\x00\x01" else ">"
        self.offset = 4

    def read(self, fmt, align):
        self.offset += (-(self.offset - 4)) % align
        value = struct.unpack_from(self.endian + fmt, self.data, self.offset)
        self.offset += struct.calcsize(self.endian + fmt)
        return value[0] if len(value) == 1 else value

    def string(self):
        length = self.read("I", 4)
        raw = bytes(self.data[self.offset:self.offset + length])
        self.offset += length
        return raw.rstrip(b"\x00").decode()

    def floats(self, count):
        self.offset += (-(self.offset - 4)) % 8
        values = np.frombuffer(self.data[self.offset:self.offset + 8 * count], dtype=self.endian + "f8").astype(np.float64)
        self.offset += 8 * count
        return values

    def header(self):
        seconds, nanoseconds = self.read("i", 4), self.read("I", 4)
        return seconds * 1_000_000_000 + nanoseconds, self.string()


def read_camera_info(payload):
    cdr = Cdr(payload)
    cdr.header()
    height, width = cdr.read("I", 4), cdr.read("I", 4)
    cdr.string()
    D = cdr.floats(cdr.read("I", 4))
    K = cdr.floats(9).reshape(3, 3)
    R = cdr.floats(9).reshape(3, 3)
    P = cdr.floats(12).reshape(3, 4)
    return {"K": K, "D": D, "R": R, "P": P, "width": width, "height": height}


def read_image(payload):
    cdr = Cdr(payload)
    stamp, _ = cdr.header()
    height, width = cdr.read("I", 4), cdr.read("I", 4)
    encoding = cdr.string().lower()
    cdr.read("B", 1)
    step, length = cdr.read("I", 4), cdr.read("I", 4)
    raw = np.frombuffer(cdr.data[cdr.offset:cdr.offset + length], dtype=np.uint8)
    if encoding in BAYER:
        return cv2.cvtColor(raw.reshape(height, step)[:, :width], BAYER[encoding])
    if encoding in ("bgr8", "rgb8"):
        image = raw.reshape(height, step)[:, :width * 3].reshape(height, width, 3)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if encoding == "rgb8" else image.copy()
    if encoding in ("mono8", "8uc1"):
        return cv2.cvtColor(raw.reshape(height, step)[:, :width], cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported encoding {encoding}")


def image_topic(camera):
    return f"/atrium/cam{camera}/image_raw"


def iter_messages(bag_path, topics):
    """Messages of the given topics from a single .mcap file or a ROS 2 bag directory (its .mcap files in name order)."""
    path = Path(bag_path)
    for file in sorted(path.glob("*.mcap")) if path.is_dir() else [path]:
        with open(file, "rb") as stream:
            yield from make_reader(stream).iter_messages(topics=topics)


def read_bag_index(bag_path, tolerance_ns, max_groups=None):
    """Return recorded CameraInfo per camera and header-stamp synchronized groups [(timestamp, {camera: stamp})]."""
    camera_infos = {}
    stamps = []
    topic_to_camera = {image_topic(c): c for c in range(CAMERAS)}
    info_topics = {f"/atrium/cam{c}/camera_info": c for c in range(CAMERAS)}
    for _, channel, message in iter_messages(bag_path, list(topic_to_camera) + list(info_topics)):
        if channel.topic in info_topics:
            camera_infos.setdefault(info_topics[channel.topic], read_camera_info(message.data))
        else:
            stamp = struct.unpack_from("<iI", message.data, 4)
            stamps.append((stamp[0] * 1_000_000_000 + stamp[1], topic_to_camera[channel.topic]))
    stamps.sort()

    buffers ={camera: deque() for camera in range(CAMERAS)}
    groups = []
    latest = None

    def drain(watermark):
        while True:
            heads = [buffer[0] for buffer in buffers.values() if buffer]
            if not heads:
                return
            anchor = min(heads)
            if anchor >= watermark:
                return
            selected = {}
            for camera, buffer in buffers.items():
                candidates = []
                for offset, stamp in enumerate(buffer):
                    if stamp - anchor > tolerance_ns:
                        break
                    if abs(stamp - anchor) <= tolerance_ns:
                        candidates.append((abs(stamp - anchor), offset, stamp))
                if candidates:
                    _, offset, stamp = min(candidates, key=lambda item: (item[0], item[1]))
                    selected[camera] = stamp
                    for _ in range(offset + 1):
                        buffer.popleft()
            groups.append((anchor, selected))

    for stamp, camera in stamps:
        buffers[camera].append(stamp)
        latest = stamp if latest is None else max(latest, stamp)
        drain(latest - tolerance_ns)
    drain(float("inf"))
    return camera_infos, groups[:max_groups]


def iter_group_images(bag_path, groups):
    """Stream images once in log order and yield (timestamp, {camera: (stamp, bgr)}) per group in timeline order."""
    wanted = {(camera, stamp): index for index, (_, frames) in enumerate(groups) for camera, stamp in frames.items()}
    missing = [len(frames) for _, frames in groups]
    images = [dict() for _ in groups]
    next_group = 0
    topic_to_camera = {image_topic(c): c for c in range(CAMERAS)}
    for _, channel, message in iter_messages(bag_path, list(topic_to_camera)):
        camera = topic_to_camera[channel.topic]
        seconds, nanoseconds = struct.unpack_from("<iI", message.data, 4)
        index = wanted.get((camera, seconds * 1_000_000_000 + nanoseconds))
        if index is None:
            continue
        images[index][camera] = (seconds * 1_000_000_000 + nanoseconds, read_image(message.data))
        missing[index] -= 1
        while next_group < len(groups) and missing[next_group] == 0:
            yield groups[next_group][0], images[next_group]
            images[next_group] = None
            next_group += 1
    for index in range(next_group, len(groups)):
        yield groups[index][0], images[index]
