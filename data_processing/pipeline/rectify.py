from pathlib import Path

import cv2
import numpy as np
import yaml


def load_cameras(config_dir, camera_infos, extrinsics_file="extrinsics_8_14.yaml"):
    """Per camera: raw->target-rectified remap, valid mask, target K and ground pose."""
    config_dir = Path(config_dir)
    extrinsics = yaml.safe_load((config_dir / extrinsics_file).read_text())["cameras"]
    cameras = {}
    for index, source in camera_infos.items():
        target = yaml.safe_load((config_dir / f"cam_{index}_intrinsic_new.yaml").read_text())
        K = np.asarray(target["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(target["distortion_coefficients"]["data"], dtype=np.float64).reshape(-1, 1)
        R = np.asarray(target["rectification_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        P = np.asarray(target["projection_matrix"]["data"], dtype=np.float64).reshape(3, 4)
        width, height = int(target["image_width"]), int(target["image_height"])
        map_x, map_y = cv2.initUndistortRectifyMap(K, D, R, P[:, :3], (width, height), cv2.CV_32FC1)
        valid = (np.isfinite(map_x) & np.isfinite(map_y) & (map_x >= 0.0) & (map_x <= source["width"] - 1.0)
                 & (map_y >= 0.0) & (map_y <= source["height"] - 1.0))
        identity = False
        if source["width"] == width and source["height"] == height:
            grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
            identity = bool(np.all(valid) and np.allclose(map_x, grid_x, rtol=0.0, atol=1e-6)
                            and np.allclose(map_y, grid_y, rtol=0.0, atol=1e-6))
        pose = np.asarray(extrinsics[f"cam{index}"]["matrix_row_major"], dtype=np.float64).reshape(3, 4)
        cameras[index] = {
            "index": index, "map_x": map_x, "map_y": map_y, "valid": valid, "identity": identity,
            "K": np.array(P[:, :3]), "R": pose[:, :3], "t": pose[:, 3], "width": width, "height": height,
        }
    return cameras


def rectify(image, camera):
    if camera["identity"]:
        return image.copy()
    output = cv2.remap(image, camera["map_x"], camera["map_y"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    output[~camera["valid"]] = 0
    return output


def pixel_to_ground(camera, uv):
    ray = camera["R"] @ np.linalg.solve(camera["K"], np.array([float(uv[0]), float(uv[1]), 1.0], dtype=np.float64))
    if float(ray[2]) >= -1e-9:
        return None
    distance = -float(camera["t"][2]) / float(ray[2])
    if distance <= 0.0:
        return None
    return (camera["t"] + distance * ray)[:2]


def sam_yaw_to_ground(global_rot_zyx, camera):
    if global_rot_zyx is None:
        return None
    angles = np.asarray(global_rot_zyx, dtype=np.float64).reshape(-1)
    if angles.size != 3 or not np.isfinite(angles).all():
        return None
    z, y, x = (float(v) for v in angles)
    rz = np.array([[np.cos(z), -np.sin(z), 0.0], [np.sin(z), np.cos(z), 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[np.cos(y), 0.0, np.sin(y)], [0.0, 1.0, 0.0], [-np.sin(y), 0.0, np.cos(y)]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(x), -np.sin(x)], [0.0, np.sin(x), np.cos(x)]])
    forward_camera = np.diag([1.0, -1.0, -1.0]) @ (rz @ ry @ rx @ np.array([0.0, 0.0, 1.0]))
    forward_ground = camera["R"] @ forward_camera
    if float(np.linalg.norm(forward_ground[:2])) <= 1e-9:
        return None
    return float(np.arctan2(forward_ground[1], forward_ground[0]))
