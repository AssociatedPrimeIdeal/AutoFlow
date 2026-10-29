"""Default independent measurements and validation for label-specific planes."""

import copy
import math
import json
import os
import stat
from pathlib import Path
import tempfile


SMOOTHING_DEFAULTS = {
    'smoothing_method': 'radius_adaptive',
    'smoothing_radius_factor': 2.0,
    'smoothing_max_displacement_ratio': 0.15,
    'smoothing_sample_step_mm': 0.5,
    'smoothing_radius_sigma_mm': 3.0,
    'smoothing_backend': 'auto',
}


def smoothing_options(payload=None):
    """Validate the shared configuration used by GUI, API and plane builders."""
    payload = dict(payload or {})
    out = {key: payload.get(key, default) for key, default in SMOOTHING_DEFAULTS.items()}
    out['smoothing_method'] = str(out['smoothing_method']).lower()
    out['smoothing_backend'] = str(out['smoothing_backend']).lower()
    if out['smoothing_method'] not in {'radius_adaptive', 'savgol', 'none'}:
        raise ValueError('smoothing_method must be radius_adaptive, savgol or none')
    if out['smoothing_backend'] not in {'auto', 'cpu', 'gpu'}:
        raise ValueError('smoothing_backend must be auto, cpu or gpu')
    for key in ('smoothing_radius_factor', 'smoothing_sample_step_mm', 'smoothing_radius_sigma_mm',
                'smoothing_max_displacement_ratio'):
        out[key] = float(out[key])
        minimum = 0 if key in {'smoothing_radius_sigma_mm', 'smoothing_max_displacement_ratio'} else 1e-12
        if not math.isfinite(out[key]) or out[key] < minimum:
            raise ValueError(f'{key} must be finite and at least {minimum:g}')
    if out['smoothing_max_displacement_ratio'] > 1:
        raise ValueError('smoothing_max_displacement_ratio must be between 0 and 1')
    return out


def default_main_path(name):
    """Choose the root side of the label's own directed vessel tree."""
    return "outflow" if str(name) in {"SV", "SMV", "SSS", "StrS", "LTS", "RTS", "LIJV", "RIJV"} else "inflow"


LABEL_LAYOUT_DEFAULTS = {
    "plane_mode": "fixed_step", "cross_section_distance": 5.0,
    "start_distance": 0.0, "end_distance": 0.0, "anchor": "center",
    "anchor_offset_mm": 5.0, "direction": "both", "spacing_mode": "fraction",
    "segmentation_filter": None, "fov_boundary_filter": None,
    "fov_boundary_max_fraction": None, "smoothing_window": None,
    "smoothing_polyorder": None, "inter_time": None,
    **{key: None for key in SMOOTHING_DEFAULTS},
}


LANDMARK_ANCHORS = {
    "label_branch", "label_reference_branch", "flow_inlet_branch", "portal_branch",
    "label_connections", "label_longest", "label_trunk", "label_reference",
    "label_level", "label_terminal", "pulmonary_split", "portal_confluence",
    "pv_first_split", "portal_trunk", "vertebral_confluence", "pca_split",
    "left_carotid_split", "right_carotid_split", "sinus_confluence",
    "aortic_origin", "arch_route",
}


def default_label_placements(name):
    """Positions on one graph branch, clipped to the target label."""
    def branch(fractions, *, inlet=False):
        return [{"anchor":"flow_inlet_branch" if inlet else "label_branch", "scope":"label",
                 "path_selection":"unique" if inlet else "longest",
                 "position":"fractions", "fractions":fractions}]
    def reference(*, fractions=None, offset=15., all_origins=False):
        spec={"anchor":"label_reference_branch", "path_selection":"all" if all_origins else "longest",
              "position":"fractions" if fractions is not None else "near"}
        spec.update(fractions=fractions) if fractions is not None else spec.update(offset_mm=offset)
        return [spec]
    if name=='AAO':return branch([.2,.5,.8])
    if name=='DAO':return branch([.1,.3,.5])
    if name in {'LICA','RICA'}:return branch([.33,.66],inlet=True)
    if name in {'LVA','RVA'}:return branch([.3,.5])
    if name=='BA':return branch([.5])
    if name in {'LIJV','RIJV'}:return branch([1/3,2/3])
    if name in {'LRA','RRA'}:return reference(fractions=[.1,.3,.5],all_origins=True)
    if name=='PV':return [{"anchor":"portal_branch","path_selection":"unique",
                          "position":"fractions","fractions":[.2,.5,.8]}]
    if name in {'MPA','LPA','RPA'}:return reference(fractions=[.3])
    if name in {'LTS','RTS','SSS','StrS'}:return reference(fractions=[.1])
    if name in {'RBCT','CCA','LBCT','SMV','SV'}:return reference()
    if name in {'ACA','LMCA','RMCA','LPCA','RPCA'}:return reference(fractions=[.1],all_origins=name=='ACA')
    return reference(offset=5.)


def default_label_plane_rules():
    names = ("AAO","MPA","ARCH","DAO","RBCT","CCA","LBCT","RPA","LPA","HA","SMA","LRA","RRA",
             "PV","SMV","SV","LICA","LVA","RVA","RICA","BA","LTS","SSS","RTS","StrS",
             "LPCA","RPCA","LMCA","RMCA","ACA","LIJV","RIJV")
    rules = {}
    for name in names:
        placements = default_label_placements(name)
        count = 0 if name in {"ARCH","HA","SMA"} else sum(
            len(spec["fractions"]) if spec["position"]=="fractions" else 1 for spec in placements)
        rules[name] = {**LABEL_LAYOUT_DEFAULTS, "plane_mode":"landmark", "plane_count":count,
                       "spacing_ratio":1/3 if count==2 else .3, "placements":placements}
    return rules


def _validate_placement(name, placement, *, allow_fallback=True):
    fields = {"anchor","position","offset_mm","fractions","level","path_selection","scope",
              "fallback","when_connected_to","otherwise_level","fraction_reference"}
    if not isinstance(placement,dict) or set(placement)-fields:
        raise ValueError(f"Invalid placement fields for {name}")
    anchor=placement.get("anchor")
    if not isinstance(anchor,str) or anchor not in LANDMARK_ANCHORS:
        raise ValueError(f"Unknown placement anchor for {name}")
    choice=placement.get("path_selection","unique")
    if not isinstance(choice,str) or choice not in {"unique","longest","all"}:
        raise ValueError(f"{name} placement path_selection must be unique, longest or all")
    if "path_selection" in placement and anchor not in {"label_level","label_terminal","label_trunk","label_reference","label_branch","label_reference_branch","flow_inlet_branch","portal_branch"}:
        raise ValueError(f"{name} path_selection applies only to label-level/interval anchors")
    position=placement.get("position")
    if not isinstance(position,str) or position not in {"near","fractions","before_end","after_start"}:
        raise ValueError(f"{name} placement position must be near, fractions, before_end or after_start")
    if position in {"before_end","after_start"} and anchor not in {"label_level","label_terminal","label_longest","label_trunk","label_branch","flow_inlet_branch"}:
        raise ValueError(f"{name} {position} requires a label path-property anchor")
    if "scope" in placement and (anchor not in {"label_level","label_terminal","label_longest","label_trunk","label_branch","flow_inlet_branch"}
                                  or not isinstance(placement["scope"],str)
                                  or placement["scope"] not in {"component","label"}):
        raise ValueError(f"{name} scope must be component or label for path-property anchors")
    if "fraction_reference" in placement and (position != "fractions" or
            anchor not in {"label_level", "label_terminal", "label_longest", "label_trunk", "label_branch", "flow_inlet_branch"} or
            placement["fraction_reference"] not in {"flow_start", "flow_end"}):
        raise ValueError(f"{name} fraction_reference requires fractions on a label path and flow_start or flow_end")
    if position=="fractions":
        values=placement.get("fractions")
        if not isinstance(values,list) or not values or any(isinstance(x,bool) or not isinstance(x,(int,float))
            or not math.isfinite(x) or not 0<x<1 for x in values):
            raise ValueError(f"{name} placement fractions must be inside (0, 1)")
        count=len(values)
    else:
        offset=placement.get("offset_mm",5.)
        if isinstance(offset,bool) or not isinstance(offset,(int,float)) or not math.isfinite(offset) or offset<0:
            raise ValueError(f"{name} placement offset_mm must be finite and non-negative")
        count=1
    if anchor=="label_level":
        level=placement.get("level",1)
        if isinstance(level,bool) or not isinstance(level,int) or level<1:
            raise ValueError(f"{name} placement level must be a positive integer (L1 = 1)")
    if "fallback" in placement or "when_connected_to" in placement or "otherwise_level" in placement:
        raise ValueError(f"{name}: fallback placements and alternate levels are no longer supported; choose one explicit anchor/level")
    return count


def normalize_label_plane_rules(value):
    """Validate complete rules; missing labels are disabled, never filled in."""
    if not isinstance(value, dict):
        raise ValueError("planes.label_rules must be an object keyed by label name or numeric id")
    defaults = default_label_plane_rules()
    result = {}
    allowed = {"plane_count", "spacing_ratio", "main_path", "placements"} | set(LABEL_LAYOUT_DEFAULTS)
    for name, raw in value.items():
        name = str(name).strip()
        if isinstance(raw, int) and not isinstance(raw, bool):
            raw = {"plane_count": raw}
        if not name or not isinstance(raw, dict):
            raise ValueError("Each label plane rule needs a non-empty label and a count or object")
        raw = dict(raw)
        # Read older saved plans without letting stale indices or PV adjacency
        # requirements override automatic root selection.
        for obsolete in ("reason", "path_index", "segment_index", "require_pv_connection"):
            raw.pop(obsolete, None)
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown plane rule fields for {name}: {sorted(unknown)}")
        # Explicit old plans retain their old layout. Only new landmark plans
        # invoke semantic anchors, and they never use the old root selector.
        rule = {**LABEL_LAYOUT_DEFAULTS, "plane_count": defaults.get(name, {}).get("plane_count", 1),
                "spacing_ratio": defaults.get(name, {}).get("spacing_ratio", .3),
                "main_path": default_main_path(name)}
        if "placements" in raw:
            rule["plane_mode"] = "landmark"
        rule.update(raw)
        count = rule["plane_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < -1:
            raise ValueError(f"{name}.plane_count must be -1 (all that fit), 0 (disabled), or a positive integer")
        # Changing the count alone also changes the standard proportional layout.
        if "plane_count" in raw and "spacing_ratio" not in raw:
            rule["spacing_ratio"] = 1 / 3 if count == 2 else 0.3
        ratio = float(rule["spacing_ratio"])
        if not math.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError(f"{name}.spacing_ratio must be finite and in (0, 1]")
        rule["spacing_ratio"] = ratio
        if rule["plane_mode"] != "landmark" and (not isinstance(rule["main_path"], str) or rule["main_path"] not in {"inflow", "outflow"}):
            raise ValueError(f"{name}.main_path must be inflow or outflow")
        for key, choices in {
            "plane_mode": {"fixed_step", "uniform", "anchored_offset", "count", "distance", "landmark"},
            "anchor": {"start", "center", "end", "junction"},
            "direction": {"toward_start", "toward_end", "both"},
            "spacing_mode": {"fraction", "distance"},
        }.items():
            if not isinstance(rule[key], str) or rule[key] not in choices:
                raise ValueError(f"{name}.{key} must be one of {sorted(choices)}")
        for key in ("cross_section_distance", "start_distance", "end_distance", "anchor_offset_mm",
                    "fov_boundary_max_fraction"):
            if rule[key] is None and key == "fov_boundary_max_fraction":
                continue
            number = float(rule[key])
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{name}.{key} must be finite and non-negative")
            if key == "cross_section_distance" and number <= 0:
                raise ValueError(f"{name}.{key} must be positive")
            if key == "fov_boundary_max_fraction" and number > 1:
                raise ValueError(f"{name}.{key} must be in [0, 1]")
            rule[key] = number
        for key in ("segmentation_filter", "fov_boundary_filter"):
            if rule[key] is not None and not isinstance(rule[key], bool):
                raise ValueError(f"{name}.{key} must be a boolean or null (inherit)")
        for key in ("smoothing_window", "smoothing_polyorder", "inter_time"):
            number = rule[key]
            if number is not None and (isinstance(number, bool) or not isinstance(number, int)
                                       or number < (0 if key == "smoothing_polyorder" else 1)):
                raise ValueError(f"{name}.{key} must be an integer or null (inherit)")
        custom_smoothing = {key: rule[key] for key in SMOOTHING_DEFAULTS if rule[key] is not None}
        if custom_smoothing:
            validated = smoothing_options(custom_smoothing)
            rule.update({key: validated[key] for key in custom_smoothing})
        if rule["plane_mode"] == "landmark":
            rule.pop("main_path", None)
            placements = rule.get("placements", default_label_placements(name))
            if not isinstance(placements, list) or not placements:
                raise ValueError(f"{name}.placements must be a non-empty array")
            total = 0
            total = sum(_validate_placement(name, placement) for placement in placements)
            if count not in (0, total):
                raise ValueError(f"{name}.plane_count must be 0 or match its {total} placement positions")
            rule["placements"] = copy.deepcopy(placements)
        result[name] = rule
    return result


def load_plane_rules_file(path):
    """Read a complete standalone plan; no default labels are merged into it."""
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Plane config must contain a JSON object: {source}")
    return normalize_label_plane_rules(payload.get("label_rules", payload))


def save_plane_rules_file(path, rules):
    """Atomically save the same standalone format consumed by CLI and GUI."""
    rules = normalize_label_plane_rules(rules)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=".plane_rules_", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump({"label_rules": rules}, handle, indent=2)
            handle.write("\n")
        mode = stat.S_IMODE(destination.stat().st_mode) if destination.exists() else 0o644
        os.chmod(temporary, mode)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return str(destination)
