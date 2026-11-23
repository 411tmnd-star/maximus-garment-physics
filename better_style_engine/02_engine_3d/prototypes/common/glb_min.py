"""glb_min.py  minimal read-only GLB accessor reader (no Draco, no sparse accessors). Prototype helper."""
import json, struct
import numpy as np
_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
def load(path):
    raw = open(path, "rb").read(); magic, ver, total = struct.unpack("<III", raw[:12]); assert magic == 0x46546C67 and ver == 2
    off = 12; js = None; binc = b""
    while off < total:
        n, t = struct.unpack("<II", raw[off:off + 8]); chunk = raw[off + 8:off + 8 + n]; off += 8 + n
        if t == 0x4E4F534A: js = json.loads(chunk)
        elif t == 0x004E4942: binc = chunk
    return js, binc
def accessor(js, binc, i):
    a = js["accessors"][i]; assert "sparse" not in a; bv = js["bufferViews"][a["bufferView"]]; dt = np.dtype(_CT[a["componentType"]]).newbyteorder("<"); nc = _NC[a["type"]]
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0); stride = bv.get("byteStride") or dt.itemsize * nc
    if stride == dt.itemsize * nc: arr = np.frombuffer(binc, dt, a["count"] * nc, start).reshape(a["count"], nc)
    else: arr = np.stack([np.frombuffer(binc, dt, nc, start + k * stride) for k in range(a["count"])])
    return arr
