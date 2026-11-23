"""placement_policy.py  explicit placement outcomes from measured support. Pure numpy/python (no Blender); the Blender layer supplies the measurements.
All thresholds are CONFIGURABLE and PROVISIONAL (defaults are design values, not validated against real mounting requirements; bridge_max_gap_m = 0.5 mm is the Step-2 provisional value).
Outcomes: ACCEPTED, ACCEPTED_WITH_CLEARANCE, FLOATING_REVIEW, SIZE_SUBSTITUTION_CANDIDATE, REJECTED. The crystal is never deformed and no filler is generated; substitution only happens when the zone enables it."""
from dataclasses import dataclass, field, asdict, replace
OUTCOMES = ('ACCEPTED', 'ACCEPTED_WITH_CLEARANCE', 'FLOATING_REVIEW', 'SIZE_SUBSTITUTION_CANDIDATE', 'REJECTED')
OUTCOME_ID = {n: i for i, n in enumerate(OUTCOMES)}
REASONS = ('NONE', 'NO_SURFACE', 'SNAP_TOO_FAR', 'AMBIGUOUS_OUTWARD', 'OCCLUDED_BY_OTHER_LAYER', 'UNSUPPORTED_BY_INTENDED_LAYER', 'EXCESS_FLOAT', 'PENETRATION', 'INCONSISTENT_NORMALS', 'OFF_PANEL',
           'CURVE_CLOSER_TO_OTHER_PANEL', 'BRIDGING', 'FLOAT_REVIEW_BAND', 'EDGE_OVERHANG', 'NORMAL_DEVIATION', 'SUBSTITUTED', 'CONTRACT_ERROR')
REASON_ID = {n: i for i, n in enumerate(REASONS)}
SIZE_ORDER = ('SS10', 'SS16', 'SS20')


@dataclass
class Thresholds:
    contact_tol_m: float = 0.0002          # a footprint point within this of the base plane counts as in contact
    accept_centre_gap_m: float = 0.0003    # centre gap up to this: proper contact (centre rests on the cloth)
    bridge_max_gap_m: float = 0.0005       # PROVISIONAL: centre gap up to this with rim support: acceptable bridging
    review_max_gap_m: float = 0.0015       # above bridge_max up to this: needs review; above: rejected as excessive floating
    min_rim_coverage: float = 0.25         # fraction of the 16 rim points in contact required for bridging
    penetration_tol_m: float = 1e-5        # footprint point more than this below the base plane = penetration
    max_normal_dev_deg: float = 10.0       # stone axis vs the evaluated local surface normal
    min_layer_support: float = 1.0         # fraction of footprint points whose first hit is the intended layer
    max_footprint_misses: int = 2          # footprint rays that find no intended surface (edge overhang) before review
    occlusion_tol_m: float = 0.0003        # another layer more than this above the intended surface along outward = occluded
    ambiguity_dot: float = 0.17            # |normal . outward| below this: direction ambiguous (~80 deg)
    mixed_normal_band: float = 0.0         # footprint faces: fraction facing outward strictly inside (band, 1-band) = inconsistent normals. 0 = ANY disagreement under one stone is reported
    def merged(self, over): bad = set(over) - set(asdict(self)); assert not bad, f'unknown thresholds {bad}'; return replace(self, **over)


@dataclass
class Measurement:
    centre_clearance_m: float; max_footprint_clearance_m: float; min_clearance_m: float; contact_coverage: float; rim_coverage: float; penetration_m: float
    normal_dev_local_deg: float; normal_dev_outward_deg: float; layer_support: float; foreign_points: int; misses: int


@dataclass
class Decision:
    outcome: str; reason: str; intended_size: str; realized_size: str; notes: list = field(default_factory=list); substitution_would_pass: list = field(default_factory=list)


def classify_gap(m, th):
    """Outcome from one measurement alone (no substitution): returns (outcome, reason)."""
    if m.penetration_m > th.penetration_tol_m: return 'REJECTED', 'PENETRATION'
    if m.layer_support < th.min_layer_support: return 'REJECTED', 'UNSUPPORTED_BY_INTENDED_LAYER'
    c = m.centre_clearance_m
    if c > th.review_max_gap_m: return 'REJECTED', 'EXCESS_FLOAT'
    if m.misses > th.max_footprint_misses: return 'FLOATING_REVIEW', 'EDGE_OVERHANG'
    if m.normal_dev_local_deg > th.max_normal_dev_deg: return 'FLOATING_REVIEW', 'NORMAL_DEVIATION'
    if c <= th.accept_centre_gap_m: return 'ACCEPTED', 'NONE'
    if c <= th.bridge_max_gap_m and m.rim_coverage >= th.min_rim_coverage: return 'ACCEPTED_WITH_CLEARANCE', 'BRIDGING'
    return 'FLOATING_REVIEW', 'FLOAT_REVIEW_BAND'


def decide(m, th, intended_size, zone_cfg=None, measure_at_size=None):
    """measure_at_size(size) -> Measurement for the same location with a smaller stone (callable supplied by the placement engine), or None.
    Substitution is APPLIED only if zone_cfg['size_substitution']['enabled'] and the smaller size is listed in allowed_to; otherwise it is only reported in `substitution_would_pass`."""
    outcome, reason = classify_gap(m, th); d = Decision(outcome, reason, intended_size, intended_size)
    if outcome == 'FLOATING_REVIEW' and reason in ('FLOAT_REVIEW_BAND', 'EDGE_OVERHANG', 'NORMAL_DEVIATION') and measure_at_size is not None:
        sub = (zone_cfg or {}).get('size_substitution', {}); allowed = [s for s in sub.get('allowed_to', []) if SIZE_ORDER.index(s) < SIZE_ORDER.index(intended_size)]
        candidates = [s for s in SIZE_ORDER[:SIZE_ORDER.index(intended_size)][::-1]]                # every smaller class, largest first, to REPORT what would pass
        for s in candidates:
            ms = measure_at_size(s)
            if ms is None: continue
            o2, r2 = classify_gap(ms, th)
            if o2 in ('ACCEPTED', 'ACCEPTED_WITH_CLEARANCE'):
                d.substitution_would_pass.append(s)
                if sub.get('enabled') and s in allowed and d.outcome != 'SIZE_SUBSTITUTION_CANDIDATE':
                    d.outcome, d.reason, d.realized_size = 'SIZE_SUBSTITUTION_CANDIDATE', 'SUBSTITUTED', s; d.notes.append(f'zone permits substitution: {intended_size} -> {s} (was {reason})')
        if d.outcome == 'FLOATING_REVIEW' and d.substitution_would_pass and not sub.get('enabled'): d.notes.append(f'smaller size would pass ({d.substitution_would_pass}); zone does not permit substitution, not applied')
    return d
