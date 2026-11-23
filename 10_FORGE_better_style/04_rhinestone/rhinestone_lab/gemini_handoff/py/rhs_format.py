"""rhs_format.py  BSRS v1.0 reference writer and reader (spec: docs/BSRS_FORMAT.md). Pure numpy/struct, no Blender.
Matrices are (N, 16) float32, column-major, glTF Y-up metres: the same packing as py/stone_matrix.pack_stone."""
import struct, zlib
import numpy as np

MAGIC = b'BSRS'; VERSION = (1, 0); HEADER_SIZE = 64; MATRIX_STRIDE = 64; META_STRIDE = 16; ENDIAN_MARKER = 0x01020304
COORD_YUP_RH, UNIT_METRE, LAYOUT_COLMAJOR_F32, META_LAYOUT_1 = 1, 1, 1, 1
SIZE_CLASSES = {0: 'SS10', 1: 'SS16', 2: 'SS20'}; SIZE_INDEX = {v: k for k, v in SIZE_CLASSES.items()}
DIAMETERS_M = (0.0028, 0.0040, 0.0048)
GEOMETRY_IDS = {1: 'chaton_v1'}; BEZEL_IDS = {0: 'none', 1: 'gold', 2: 'silver'}
ZONES = {0: 'unspecified', 1: 'collar', 2: 'sleeve_border', 3: 'back_script', 4: 'elephant', 5: 'outseam', 6: 'test'}
FLAG_LIFTED, FLAG_FLOATING, FLAG_LOW_CONFIDENCE, FLAG_ENDPOINT = 1, 2, 4, 8; FLAG_MASK = 0x000F
HDR = struct.Struct('<4sHHIIIIIBBBB3fIIIII'); META = np.dtype([('size_class', 'u1'), ('r0', 'u1'), ('geometry_id', '<u2'), ('bezel_id', '<u2'), ('zone_id', '<u2'), ('flags', '<u2'), ('r1', '<u2'), ('sample_index', '<u4')])
assert HDR.size == 64 and META.itemsize == 16
MAX_RECORDS = (2 ** 31) // (MATRIX_STRIDE + META_STRIDE)
# ---- v1.1 placement extension (docs/BSRS_FORMAT.md section 6)
SUPPORTED_MINOR = 1; EXT_FLAG = 1; EXT_STRIDE = 16
OUTCOMES = ('ACCEPTED', 'ACCEPTED_WITH_CLEARANCE', 'FLOATING_REVIEW', 'SIZE_SUBSTITUTION_CANDIDATE', 'REJECTED')
REASONS = ('NONE', 'NO_SURFACE', 'SNAP_TOO_FAR', 'AMBIGUOUS_OUTWARD', 'OCCLUDED_BY_OTHER_LAYER', 'UNSUPPORTED_BY_INTENDED_LAYER', 'EXCESS_FLOAT', 'PENETRATION', 'INCONSISTENT_NORMALS', 'OFF_PANEL',
           'CURVE_CLOSER_TO_OTHER_PANEL', 'BRIDGING', 'FLOAT_REVIEW_BAND', 'EDGE_OVERHANG', 'NORMAL_DEVIATION', 'SUBSTITUTED', 'CONTRACT_ERROR')
EXT = struct.Struct('<BBHHHHHBBH'); assert EXT.size == 16


def _crc(b): return zlib.crc32(b) & 0xFFFFFFFF


def _header_bytes(count, payload_crc, total, *, major=VERSION[0], minor=VERSION[1], endian=ENDIAN_MARKER, header_size=HEADER_SIZE, ids=(COORD_YUP_RH, UNIT_METRE, LAYOUT_COLMAJOR_F32, META_LAYOUT_1),
                  diameters=DIAMETERS_M, file_flags=0, mat_off=HEADER_SIZE, meta_off=None, reserved=0, magic=MAGIC):
    meta_off = ((HEADER_SIZE + MATRIX_STRIDE * count) & 0xFFFFFFFF) if meta_off is None else meta_off
    raw = HDR.pack(magic, major, minor, header_size, count & 0xFFFFFFFF, mat_off, meta_off, endian, *ids, *diameters, file_flags, payload_crc, 0, total, reserved)
    return raw[:52] + struct.pack('<I', _crc(raw)) + raw[56:]


def write_bsrs(matrices, size_class, bezel_id=None, zone_id=None, flags=None, geometry_id=None, sample_index=None, placement=None):
    """Serialise. matrices (N,16) float32 column-major; size_class (N,) ints 0..2 (or 'SS10'.. strings). Returns bytes."""
    m = np.ascontiguousarray(np.asarray(matrices, dtype='<f4').reshape(-1, 16)); n = len(m)
    sc = np.array([SIZE_INDEX[s] if isinstance(s, str) else int(s) for s in size_class], dtype='u1')
    if len(sc) != n: raise ValueError('size_class length != matrix count')
    meta = np.zeros(n, META); meta['size_class'] = sc
    for key, val, default in (('bezel_id', bezel_id, 0), ('zone_id', zone_id, 0), ('flags', flags, 0), ('geometry_id', geometry_id, 1), ('sample_index', sample_index, None)):
        meta[key] = (np.arange(n) if default is None else default) if val is None else np.broadcast_to(np.asarray(val), (n,))
    payload = m.tobytes() + meta.tobytes()
    if placement is None:
        total = HEADER_SIZE + len(payload); return _header_bytes(n, _crc(payload), total) + payload
    ext = _pack_ext(placement, n); ext_off = HEADER_SIZE + len(payload); payload += ext; total = HEADER_SIZE + len(payload)       # v1.1: minor 1, EXT flag, header[60] = ext_offset
    return _header_bytes(n, _crc(payload), total, minor=1, file_flags=EXT_FLAG, reserved=ext_off) + payload


def _idx(v, names): return names.index(v) if isinstance(v, str) else int(v)


def _pack_ext(pl, n):
    strings = []; table = {}
    def sid(x):
        if x not in table: table[x] = len(strings); strings.append(x)
        return table[x]
    for key in ('intended_size', 'outcome', 'reason', 'panel', 'surface', 'layer', 'curve'):
        if len(pl[key]) != n: raise ValueError(f'placement[{key}] length != stone count')
    rec = bytearray()
    for i in range(n):
        isz = SIZE_INDEX[pl['intended_size'][i]] if isinstance(pl['intended_size'][i], str) else int(pl['intended_size'][i])
        rec += EXT.pack(isz, _idx(pl['outcome'][i], OUTCOMES), _idx(pl['reason'][i], REASONS), sid(pl['panel'][i]), sid(pl['surface'][i]), sid(pl['layer'][i]), sid(pl['curve'][i]), int(pl.get('collisions', [0] * n)[i]), int(pl.get('collision_severity', [0] * n)[i]), 0)
    if len(strings) > 65535: raise ValueError('too many distinct id strings')
    tab = struct.pack('<I', len(strings))
    for x in strings:
        b = x.encode('utf-8'); tab += struct.pack('<H', len(b)) + b
    tab += b'\0' * (-len(tab) % 4); return bytes(rec) + tab


def _err(errors, code, msg, **kw): errors.append({'code': code, 'message': msg, **kw})


def _read_ext(b, off, count, total, errors, res):
    ext = np.frombuffer(b, np.dtype([('intended', 'u1'), ('outcome', 'u1'), ('reason', '<u2'), ('panel', '<u2'), ('surface', '<u2'), ('layer', '<u2'), ('curve', '<u2'), ('ncoll', 'u1'), ('sev', 'u1'), ('r', '<u2')]), count, off)
    tp = off + count * EXT_STRIDE; (ns,) = struct.unpack_from('<I', b, tp); p = tp + 4; strings = []
    for _ in range(ns):
        if p + 2 > total: _err(errors, 'E_EXT_STRINGS', 'string table truncated'); return
        (ln,) = struct.unpack_from('<H', b, p); p += 2
        if p + ln > total: _err(errors, 'E_EXT_STRINGS', 'string table truncated'); return
        try: strings.append(b[p:p + ln].decode('utf-8'))
        except UnicodeDecodeError: _err(errors, 'E_EXT_STRINGS', 'invalid utf-8'); return
        p += ln
    if total - p != (-(p - tp) % 4) or any(b[p:total]): _err(errors, 'E_EXT_STRINGS', 'string table padding/length mismatch'); return
    def idx(mask): w = np.flatnonzero(mask); return {'count': int(len(w)), 'first_indices': w[:5].tolist()}
    for name, arr, lim, code in (('intended', ext['intended'], 3, 'E_EXT_SIZE'), ('outcome', ext['outcome'], len(OUTCOMES), 'E_EXT_OUTCOME'), ('reason', ext['reason'], len(REASONS), 'E_EXT_REASON')):
        bad = arr >= lim
        if bad.any(): _err(errors, code, f'{name} id out of range', **idx(bad))
    for name in ('panel', 'surface', 'layer', 'curve'):
        bad = ext[name] >= ns
        if bad.any(): _err(errors, 'E_EXT_INDEX', f'{name} string index out of range', **idx(bad))
    if (ext['r'] != 0).any(): _err(errors, 'E_EXT_RESERVED', 'ext reserved non-zero', **idx(ext['r'] != 0))
    if (ext['sev'] > 3).any(): _err(errors, 'E_EXT_SEVERITY', 'collision severity > 3', **idx(ext['sev'] > 3))
    res['placement'] = {'ext': ext, 'strings': strings}


def read_bsrs(buf):
    """Parse + validate. Returns dict(ok, errors, warnings, header, matrices (N,16) f32, meta (structured array)). Never raises on bad input."""
    errors, warnings = [], []; res = {'ok': False, 'errors': errors, 'warnings': warnings, 'header': None, 'matrices': None, 'meta': None}
    b = bytes(buf)
    if len(b) < HEADER_SIZE: _err(errors, 'E_TOO_SHORT', f'{len(b)} bytes, header needs {HEADER_SIZE}'); return res
    (magic, major, minor, hsize, count, moff, eoff, endian, coord, unit, lay, mlay, d0, d1, d2, fflags, pcrc, hcrc, total, reserved) = HDR.unpack_from(b, 0)
    h = dict(version=(major, minor), header_size=hsize, record_count=count, matrix_offset=moff, meta_offset=eoff, coord_id=coord, unit_id=unit, matrix_layout_id=lay, meta_layout_id=mlay, diameters_m=(d0, d1, d2), file_flags=fflags, total_size=total)
    res['header'] = h
    if magic != MAGIC: _err(errors, 'E_BAD_MAGIC', f'magic {magic!r}'); return res
    if endian != ENDIAN_MARKER:
        _err(errors, 'E_ENDIAN', 'byte-swapped file (marker 0x%08X)' % endian if endian == 0x04030201 else f'endian marker 0x{endian:08X}'); return res
    if major != VERSION[0]: _err(errors, 'E_VERSION_MAJOR', f'major {major}, supported {VERSION[0]}'); return res
    if hsize != HEADER_SIZE: _err(errors, 'E_HEADER_SIZE', f'header_size {hsize}'); return res
    hb = b[:52] + b'\0\0\0\0' + b[56:64]
    if _crc(hb) != hcrc: _err(errors, 'E_HEADER_CRC', 'header checksum mismatch'); return res
    if minor > SUPPORTED_MINOR: warnings.append(f'minor version {minor} is newer than supported {SUPPORTED_MINOR}')
    if (coord, unit, lay, mlay) != (COORD_YUP_RH, UNIT_METRE, LAYOUT_COLMAJOR_F32, META_LAYOUT_1): _err(errors, 'E_UNKNOWN_ID', f'coord {coord} unit {unit} layout {lay} meta {mlay}'); return res
    has_ext = bool(fflags & EXT_FLAG)
    if (fflags & ~EXT_FLAG) or (has_ext and minor < 1): _err(errors, 'E_FLAGS', f'file_flags 0x{fflags:X} (minor {minor})'); return res
    if not np.allclose((d0, d1, d2), DIAMETERS_M, rtol=0, atol=1e-9): _err(errors, 'E_DIAMETER_TABLE', f'{(d0, d1, d2)}'); return res
    if reserved != 0 and not has_ext: _err(errors, 'E_RESERVED', 'header reserved field non-zero'); return res
    base_end = HEADER_SIZE + count * (MATRIX_STRIDE + META_STRIDE)
    if count > MAX_RECORDS or (not has_ext and base_end != total) or (has_ext and total < base_end + count * EXT_STRIDE + 4): _err(errors, 'E_COUNT_OVERFLOW', f'count {count} does not match total_size {total}'); return res
    if has_ext and reserved != base_end: _err(errors, 'E_EXT_OFFSET', f'ext_offset {reserved}, expected {base_end}'); return res
    if len(b) != total: _err(errors, 'E_SIZE_MISMATCH', f'buffer {len(b)} bytes, header says {total}'); return res
    if moff != HEADER_SIZE or eoff != HEADER_SIZE + MATRIX_STRIDE * count: _err(errors, 'E_OFFSETS', f'matrix {moff} meta {eoff}'); return res
    if _crc(b[HEADER_SIZE:]) != pcrc: _err(errors, 'E_PAYLOAD_CRC', 'payload checksum mismatch'); return res
    m = np.frombuffer(b, '<f4', count * 16, moff).reshape(count, 16); meta = np.frombuffer(b, META, count, eoff)
    res['matrices'], res['meta'] = m, meta
    if has_ext: _read_ext(b, base_end, count, total, errors, res)
    def idx(mask): w = np.flatnonzero(mask); return {'count': int(len(w)), 'first_indices': w[:5].tolist()}
    bad = (meta['r0'] != 0) | (meta['r1'] != 0)
    if bad.any(): _err(errors, 'E_RESERVED', 'reserved meta field non-zero', **idx(bad))
    nf = ~np.isfinite(m).all(1)
    if nf.any(): _err(errors, 'E_NONFINITE', 'NaN/Inf in matrix', **idx(nf))
    ok = ~nf; mm = np.where(nf[:, None], np.eye(4, dtype='f4').reshape(-1)[None, :].repeat(count, 0) if count else m, m).astype(np.float64)
    row = (mm[:, [3, 7, 11, 15]] != np.array([0, 0, 0, 1.0])).any(1) & ok
    if row.any(): _err(errors, 'E_MATRIX_ROW', 'bottom row is not (0,0,0,1)', **idx(row))
    X, Y, Z = mm[:, 0:3], mm[:, 4:7], mm[:, 8:11]; lx, ly, lz = (np.linalg.norm(v, axis=1) for v in (X, Y, Z))
    with np.errstate(all='ignore'):
        dots = np.stack([(X * Y).sum(1) / (lx * ly), (Y * Z).sum(1) / (ly * lz), (X * Z).sum(1) / (lx * lz)], 1); hand = (np.cross(X, Y) * Z).sum(1)
    basis = ((np.abs(dots) > 1e-4).any(1) | (hand <= 0) | ~np.isfinite(dots).all(1)) & ok
    if basis.any(): _err(errors, 'E_MATRIX_BASIS', 'basis not orthogonal or left-handed', **idx(basis))
    sc = meta['size_class']; badc = sc > 2
    if badc.any(): _err(errors, 'E_SIZE_CLASS', 'size_class not in {0,1,2}', **idx(badc))
    d = np.array(DIAMETERS_M)[np.minimum(sc, 2)]; sc_bad = (((np.abs(lx - d) > d * 1e-3) | (np.abs(ly - d) > d * 1e-3) | (np.abs(lz - d) > d * 1e-3)) & ok & ~badc)
    if sc_bad.any(): _err(errors, 'E_SCALE_CLASS', 'matrix scale does not match size class', **idx(sc_bad))
    g = ~np.isin(meta['geometry_id'], list(GEOMETRY_IDS))
    if g.any(): _err(errors, 'E_GEOMETRY_ID', 'unregistered geometry_id', **idx(g))
    z = ~np.isin(meta['bezel_id'], list(BEZEL_IDS))
    if z.any(): _err(errors, 'E_BEZEL_ID', 'unregistered bezel_id', **idx(z))
    f = (meta['flags'] & ~np.uint16(FLAG_MASK)) != 0
    if f.any(): _err(errors, 'E_STONE_FLAGS', 'unknown stone flag bits', **idx(f))
    res['ok'] = not errors; return res
