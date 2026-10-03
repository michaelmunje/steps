"""Recover interleaved DeepStream CSV records without inventing lost values.

Output keeps source record order and repeated observations. Missing scores are
nullable, and unknown orientations use empty yaw with orient_valid=0. Conflicting
same-stamp IDs are retained for a separate, explicit snapshot assembly policy.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import re

HEADER = 'stamp_ns,track_id,x,y,yaw_rad,orient_valid,score'
START = re.compile(r'1787\d{15},\d+,')
XY = re.compile(r'-?\d+\.\d{4}')
YAW = re.compile(r'-?\d+\.\d{5}')
SCORE = re.compile(r'(?:0\.\d{3}|1\.000)')
STAMP = re.compile(r'1787\d{15}')


def complete(fields):
    return (len(fields) == 7 and STAMP.fullmatch(fields[0]) and fields[1].isdigit()
            and XY.fullmatch(fields[2]) and XY.fullmatch(fields[3])
            and ((fields[4] == '' and fields[5] == '0')
                 or (YAW.fullmatch(fields[4]) and fields[5] == '1'))
            and SCORE.fullmatch(fields[6]))


def valid_repaired(fields):
    return bool(complete(fields) or (len(fields) == 7 and fields[6] == ''
                                   and complete([*fields[:6], '0.000'])))


def repair_bytes(data: bytes):
    lines = data.decode('utf-8').splitlines()
    if not lines or lines[0] != HEADER:
        raise ValueError('Unexpected CSV header')
    segments, quarantine, donors = [], [], collections.defaultdict(set)
    for line_number, line in enumerate(lines[1:], 2):
        starts = list(START.finditer(line))
        if not starts:
            if line:
                quarantine.append({'line':line_number, 'column':0, 'reason':'no_complete_record_header', 'text':line})
            continue
        if starts[0].start():
            quarantine.append({'line':line_number, 'column':0, 'reason':'truncated_prefix_before_record',
                               'text':line[:starts[0].start()]})
        for j, match in enumerate(starts):
            end = starts[j+1].start() if j+1 < len(starts) else len(line)
            text = line[match.start():end]
            fields = text.split(',')
            segment = {'line':line_number, 'column':match.start(), 'text':text, 'fields':fields}
            segments.append(segment)
            if complete(fields):
                donors[tuple(fields[:2])].add(text)
    rows, repairs, counts = [], [], collections.Counter()
    for segment in segments:
        fields, original = segment['fields'], segment['text']
        if complete(fields):
            output, kind = fields, 'intact_complete'
        else:
            matches = [text for text in donors[tuple(fields[:2])] if text.startswith(original)]
            if len(matches) == 1:
                output, kind = matches[0].split(','), 'restored_exact_same_stamp_id_donor'
            elif len(fields) >= 4 and XY.fullmatch(fields[2]) and XY.fullmatch(fields[3]):
                # Complete position is recoverable. Missing/truncated metadata
                # is not guessed from neighboring people, times or confidence.
                known_yaw = len(fields) >= 6 and YAW.fullmatch(fields[4]) and fields[5] == '1'
                yaw, orient = (fields[4], '1') if known_yaw else ('', '0')
                score = fields[6] if len(fields) == 7 and SCORE.fullmatch(fields[6]) else ''
                output, kind = [*fields[:4], yaw, orient, score], 'position_retained_unknown_metadata'
            else:
                counts['quarantined_incomplete_position'] += 1
                quarantine.append({k:segment[k] for k in ('line','column','text')} |
                                  {'reason':'incomplete_position_no_unique_same_stamp_donor'})
                continue
        assert valid_repaired(output)
        rows.append(output)
        counts[kind] += 1
        if kind != 'intact_complete':
            repairs.append({'output_row':len(rows)+1, 'line':segment['line'], 'column':segment['column'],
                            'method':kind, 'original_fragment':original, 'repaired_record':','.join(output)})
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(HEADER.split(',')); writer.writerows(rows)
    output_bytes = stream.getvalue().encode()
    # A multiset assertion proves all complete input records were retained,
    # including conflicting duplicates; they are not silently discarded.
    intact = collections.Counter(s['text'] for s in segments if complete(s['fields']))
    output_counts = collections.Counter(','.join(r) for r in rows)
    assert not (intact-output_counts)
    key_counts = collections.Counter(tuple(r[:2]) for r in rows)
    payloads = collections.defaultdict(set)
    for row in rows: payloads[tuple(row[:2])].add(tuple(row[2:]))
    report = {'schema_version':1, 'source_sha256':hashlib.sha256(data).hexdigest(),
              'repaired_sha256':hashlib.sha256(output_bytes).hexdigest(),
              'physical_input_data_lines':len(lines)-1, 'detected_record_segments':len(segments),
              'output_rows':len(rows), 'counts':dict(counts), 'quarantined_fragments':len(quarantine),
              'missing_confidence_rows':sum(r[6]=='' for r in rows),
              'orientation_known_rows':sum(r[5]=='1' for r in rows),
              'repeated_stamp_id_extra_rows':sum(n-1 for n in key_counts.values()),
              'conflicting_stamp_id_groups':sum(len(v)>1 for v in payloads.values()),
              'complete_input_records_preserved':True,
              'record_order_policy':'Original line and byte-column order; repeated observations retained',
              'unknown_policy':'Empty score means lost/unknown confidence; empty yaw with flag0 means unavailable heading',
              'limits':['Irrecoverable timestamp/coordinate fragments are quarantined, not fabricated',
                        'The output is an observation event table; conflicting same-stamp IDs need explicit assembly before metrics'],
              'repairs':repairs, 'quarantine':quarantine}
    return output_bytes, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.audit.exists():
        raise ValueError('Output/audit path exists; preserve it and choose a new path')
    before = args.input.stat()
    source = args.input.read_bytes()
    after = args.input.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('Input changed during capture')
    output, report = repair_bytes(source)
    report.update(source_path=str(args.input.absolute()),
                  source_signature={'size':after.st_size,'mtime_ns':after.st_mtime_ns,'inode':after.st_ino})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f: f.write(output)
    with args.audit.open('x') as f: json.dump(report,f,indent=2); f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('repairs','quarantine')},indent=2))


if __name__ == '__main__':
    main()
