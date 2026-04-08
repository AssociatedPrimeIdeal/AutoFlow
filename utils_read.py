import re
import json
import numpy as np

def _parse_scalar_value(s):
    s = s.strip()
    if s in {"--", ""}:
        return np.nan
    m = re.match(r"^-?\d+(?:\.\d+)?", s)
    if m:
        return float(m.group())
    return s

def parse_cvi_full(path):
    text = open(path, encoding="utf-8", errors="ignore").read()
    parts = re.split(r"Flow\s+(\d+)\s*-\s*完成报告", text)
    out = {}
    for i in range(1, len(parts), 2):
        flow_id = int(parts[i])
        block = parts[i + 1]
        plane = {"flow_id": flow_id}
        lines = [x.rstrip() for x in block.splitlines()]
        table_idx = None
        for j, line in enumerate(lines):
            if re.match(r"^\s*时间\s+面积\s+正向面积\s+负向面积\s+流速\s+正向流速\s+负向流速\s+平均速度\s+最大速度\s+最小速度\s+标准偏差速度\s*$", line):
                table_idx = j
                break
        head_lines = lines[:table_idx] if table_idx is not None else lines
        for line in head_lines:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(.*?)\s+(-?\d+(?:\.\d+)?|--)(?:\s+\S+)?\s*$", line)
            if m:
                k = m.group(1).strip()
                v = m.group(2).strip()
                plane[k] = _parse_scalar_value(v)
        if table_idx is not None and table_idx + 1 < len(lines):
            headers = re.split(r"\s+", lines[table_idx].strip())
            data = []
            for line in lines[table_idx + 2:]:
                vals = re.findall(r"-?\d+(?:\.\d+)?", line)
                if len(vals) == len(headers):
                    data.append([float(x) for x in vals])
                elif data:
                    break
            if data:
                arr = np.asarray(data, dtype=float)
                for k, h in enumerate(headers):
                    plane[h] = arr[:, k]
        out[f"flow_{flow_id}"] = plane
    return out

def parse_json_full(path):
    data = json.load(open(path, encoding="utf-8"))
    out = {}
    for i, d in enumerate(data):
        plane = {}
        for k, v in d.items():
            if isinstance(v, list):
                if len(v) == 0:
                    plane[k] = np.array([])
                elif all(isinstance(x, (int, float, np.integer, np.floating)) for x in v):
                    plane[k] = np.asarray(v, dtype=float)
                else:
                    plane[k] = np.asarray(v, dtype=object)
            else:
                plane[k] = v
        out[f"plane_{i}"] = plane
    return out