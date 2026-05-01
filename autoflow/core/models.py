import copy, json, uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ObjectKind(Enum):
    SEGMENTATION = "Segmentation"
    SKELETON = "Skeleton"
    GRAPH = "Graph"
    BRANCH = "Branch"
    PLANE = "Plane"
    FLOW = "Flow"
    METRIC = "Metric"
    AUX = "Aux"


class StepId(Enum):
    GENERATE_SKELETON = "step_skeleton"
    EDIT_SKELETON = "step_edit_skeleton"
    GENERATE_GRAPH = "step_graph"
    EDIT_GRAPH = "step_edit_graph"
    GENERATE_PLANES = "step_planes"
    EDIT_PLANES = "step_edit_planes"
    GENERATE_STREAMLINES = "step_streamlines"
    PLANE_STREAMLINES = "step_plane_streamlines"
    COMPUTE_PLANE_METRICS = "step_plane_metrics"
    COMPUTE_DERIVED_METRICS = "step_derived_metrics"

    @property
    def label(self):
        return {
            StepId.GENERATE_SKELETON: "Generate Skeleton",
            StepId.EDIT_SKELETON: "Edit Skeleton",
            StepId.GENERATE_GRAPH: "Generate Graph",
            StepId.EDIT_GRAPH: "Edit Graph",
            StepId.GENERATE_PLANES: "Generate Planes",
            StepId.EDIT_PLANES: "Edit Planes",
            StepId.GENERATE_STREAMLINES: "Generate Streamlines",
            StepId.PLANE_STREAMLINES: "Plane Streamlines",
            StepId.COMPUTE_PLANE_METRICS: "Calculate && Save Metrics",
            StepId.COMPUTE_DERIVED_METRICS: "WSS / TKE",
        }[self]

    @staticmethod
    def top_row_steps():
        return [
            StepId.GENERATE_SKELETON,
            StepId.GENERATE_GRAPH,
            StepId.GENERATE_PLANES,
            StepId.COMPUTE_PLANE_METRICS,
        ]

    @staticmethod
    def bottom_row_steps():
        return [
            StepId.EDIT_SKELETON,
            StepId.EDIT_GRAPH,
        ]

    @staticmethod
    def extra_row_steps():
        return [
            StepId.GENERATE_STREAMLINES,
            StepId.PLANE_STREAMLINES,
            StepId.COMPUTE_DERIVED_METRICS,
        ]


@dataclass
class PreprocessParams:
    def to_dict(self):
        return {}

    @staticmethod
    def from_dict(d):
        return PreprocessParams()


@dataclass
class SkeletonParams:
    remove_small_cc: bool = False
    min_cc_volume_mm3: float = 50.0
    do_closing: bool = True
    do_opening: bool = False
    gaussian_sigma: float = 0.5
    gaussian_enabled: bool = True

    def to_dict(self):
        return {
            "remove_small_cc": self.remove_small_cc,
            "min_cc_volume_mm3": self.min_cc_volume_mm3,
            "do_closing": self.do_closing,
            "do_opening": self.do_opening,
            "gaussian_sigma": self.gaussian_sigma,
            "gaussian_enabled": self.gaussian_enabled,
        }

    @staticmethod
    def from_dict(d):
        if "keep_largest_cc" in d and "remove_small_cc" not in d:
            d["remove_small_cc"] = d["keep_largest_cc"]
        return SkeletonParams(
            remove_small_cc=bool(d.get("remove_small_cc", False)),
            min_cc_volume_mm3=float(d.get("min_cc_volume_mm3", 50.0)),
            do_closing=bool(d.get("do_closing", True)),
            do_opening=bool(d.get("do_opening", False)),
            gaussian_sigma=float(d.get("gaussian_sigma", 0.5)),
            gaussian_enabled=bool(d.get("gaussian_enabled", True)))


@dataclass
class PlaneGenerationParams:
    use_center_plane: bool = True
    cross_section_distance: float = 20.0
    start_distance: float = 5.0
    end_distance: float = 0.0
    smoothing_window: int = 15
    smoothing_polyorder: int = 2
    inter_time: int = 10

    def to_dict(self):
        return {
            "use_center_plane": bool(self.use_center_plane),
            "cross_section_distance": self.cross_section_distance,
            "start_distance": self.start_distance,
            "end_distance": self.end_distance,
            "smoothing_window": int(self.smoothing_window),
            "smoothing_polyorder": int(self.smoothing_polyorder),
            "inter_time": int(self.inter_time),
        }

    @staticmethod
    def from_dict(d):
        return PlaneGenerationParams(
            use_center_plane=bool(d.get("use_center_plane", True)),
            cross_section_distance=float(d.get("cross_section_distance", 20.0)),
            start_distance=float(d.get("start_distance", 5.0)),
            end_distance=float(d.get("end_distance", 0.0)),
            smoothing_window=int(d.get("smoothing_window", 15)),
            smoothing_polyorder=int(d.get("smoothing_polyorder", 3)),
            inter_time=int(d.get("inter_time", 10)),
        )


@dataclass
class StreamlineParams:
    seed_ratio: float = 0.02
    max_steps: int = 2000
    min_seeds: int = 50
    terminal_speed: float = 0.01
    rng_seed: int = 0

    def to_dict(self):
        return {
            "seed_ratio": self.seed_ratio,
            "max_steps": self.max_steps,
            "min_seeds": self.min_seeds,
            "terminal_speed": self.terminal_speed,
            "rng_seed": self.rng_seed,
        }

    @staticmethod
    def from_dict(d):
        return StreamlineParams(
            seed_ratio=float(d.get("seed_ratio", 0.02)),
            max_steps=int(d.get("max_steps", 2000)),
            min_seeds=50,
            terminal_speed=float(d.get("terminal_speed", 0.01)),
            rng_seed=int(d.get("rng_seed", 0)),
        )


@dataclass
class DerivedMetricsParams:
    smoothing_iteration: int = 200
    viscosity: float = 4.0
    inward_distance: float = 0.6
    parabolic_fitting: bool = True
    no_slip_condition: bool = True
    step_size: int = 5
    tube_radius: float = 0.1
    rho: float = 1060.0
    use_multithread: bool = False

    def to_dict(self):
        return {
            "smoothing_iteration": self.smoothing_iteration,
            "viscosity": self.viscosity,
            "inward_distance": self.inward_distance,
            "parabolic_fitting": self.parabolic_fitting,
            "no_slip_condition": self.no_slip_condition,
            "step_size": self.step_size,
            "tube_radius": self.tube_radius,
            "rho": self.rho,
            "use_multithread": self.use_multithread,
        }

    @staticmethod
    def from_dict(d):
        return DerivedMetricsParams(
            smoothing_iteration=int(d.get("smoothing_iteration", 200)),
            viscosity=float(d.get("viscosity", 4.0)),
            inward_distance=float(d.get("inward_distance", 0.6)),
            parabolic_fitting=bool(d.get("parabolic_fitting", True)),
            no_slip_condition=bool(d.get("no_slip_condition", True)),
            step_size=int(d.get("step_size", 5)),
            tube_radius=float(d.get("tube_radius", 0.1)),
            rho=float(d.get("rho", 1060.0)),
            use_multithread=bool(d.get("use_multithread", False)),
        )


@dataclass
class PathsState:
    segmask_path: str = ""
    flow_path: str = ""
    workspace_path: str = ""
    output_dir: str = ""


@dataclass
class PipelineFlags:
    completed: Dict[str, bool] = field(default_factory=dict)
    skipped: Dict[str, bool] = field(default_factory=dict)

    def mark_done(self, step: StepId, skipped: bool = False):
        self.completed[step.value] = True
        self.skipped[step.value] = bool(skipped)

    def is_done(self, step: StepId) -> bool:
        return bool(self.completed.get(step.value, False))

    def reset(self):
        self.completed.clear()
        self.skipped.clear()


@dataclass
class SceneObject:
    uid: str
    name: str
    kind: ObjectKind
    data_key: str
    visible: bool = True
    opacity: float = 1.0
    color: str = "white"
    scalars: Optional[str] = None
    cmap: str = "turbo"
    clim: Optional[Tuple[float, float]] = None
    point_size: int = 8
    line_width: int = 2
    tube_radius: float = 0.0
    show_scalar_bar: bool = False
    scalar_bar_title: Optional[str] = None
    dynamic: bool = False
    actor: Any = None
    label_actor: Any = None


@dataclass
class PlaneData:
    center: np.ndarray
    normal: np.ndarray
    label: int = 1
    path_index: int = 0
    distance: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphData:
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=float))
    edges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))


@dataclass
class DerivedResults:
    plane_metrics: List[Dict[str, Any]] = field(default_factory=list)
    plane_qc: Dict[str, Any] = field(default_factory=dict)
    wss_surfaces: List[Any] = field(default_factory=list)
    wss_volume: Optional[np.ndarray] = None
    tke_volume: Any = None
    tke_array: Optional[np.ndarray] = None
    streamlines: List[Any] = field(default_factory=list)
    pixelwise_export: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workspace:
    paths: PathsState = field(default_factory=PathsState)
    pipeline: PipelineFlags = field(default_factory=PipelineFlags)
    preprocess_params: PreprocessParams = field(default_factory=PreprocessParams)
    skeleton_params: SkeletonParams = field(default_factory=SkeletonParams)
    plane_gen_params: PlaneGenerationParams = field(default_factory=PlaneGenerationParams)
    streamline_params: StreamlineParams = field(default_factory=StreamlineParams)
    derived_params: DerivedMetricsParams = field(default_factory=DerivedMetricsParams)

    resolution: np.ndarray = field(default_factory=lambda: np.array([1., 1., 1.]))
    origin: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0.]))
    spatial_order: List[str] = field(default_factory=lambda: ["FH", "AP", "LR"])
    venc_order: List[str] = field(default_factory=lambda: ["FH", "AP", "LR"])
    venc: np.ndarray = field(default_factory=lambda: np.array([1., 1., 1.]))
    rr: float = 1000.0

    segmask_raw: Optional[np.ndarray] = None
    segmask_labels: Optional[np.ndarray] = None
    segmask_binary: Optional[np.ndarray] = None
    segmask_3d: Optional[np.ndarray] = None

    mag_raw: Optional[np.ndarray] = None

    skeleton_points: Optional[np.ndarray] = None
    skeleton_mask: Optional[np.ndarray] = None

    graph: GraphData = field(default_factory=GraphData)
    branch_labels: Optional[np.ndarray] = None
    centerline_paths: List[np.ndarray] = field(default_factory=list)
    centerline_node_paths: List[List[int]] = field(default_factory=list)
    centerline_paths_smooth: List[np.ndarray] = field(default_factory=list)
    path_info: List[Dict[str, Any]] = field(default_factory=list)
    forks: List[Dict[str, Any]] = field(default_factory=list)
    planes: List[PlaneData] = field(default_factory=list)

    flow_raw: Optional[np.ndarray] = None
    streamline_seeds: Optional[np.ndarray] = None
    streamline_cache: Dict[int, Any] = field(default_factory=dict)
    streamline_active: bool = False
    plane_streamline_cache: Dict[int, Any] = field(default_factory=dict)
    plane_streamline_active: bool = False
    plane_streamline_plane_idx: int = -1
    derived: DerivedResults = field(default_factory=DerivedResults)

    scene_objects: Dict[str, SceneObject] = field(default_factory=dict)
    current_t: int = 0
    data_loaded: bool = False

    ortho_cursor: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0], dtype=int))
    selected_path_index: int = -1

    def time_count(self):
        if self.flow_raw is not None and self.flow_raw.ndim == 5:
            return int(self.flow_raw.shape[3])
        if self.segmask_raw is not None and self.segmask_raw.ndim == 4:
            return int(self.segmask_raw.shape[3])
        return 1

    def has_flow(self):
        return self.flow_raw is not None

    def unique_labels(self):
        if self.segmask_raw is None:
            return []
        return sorted(int(x) for x in np.unique(self.segmask_raw) if x != 0)

    def add_object(self, name, kind, data_key, **kw):
        uid = str(uuid.uuid4())
        self.scene_objects[uid] = SceneObject(uid=uid, name=name, kind=kind, data_key=data_key, **kw)
        return uid

    def remove_object(self, uid):
        return self.scene_objects.pop(uid, None)

    def remove_object_by_data_key(self, data_key):
        to_del = [u for u, o in self.scene_objects.items() if o.data_key == data_key]
        return [self.scene_objects.pop(u) for u in to_del]

    def remove_objects_by_prefix(self, prefix):
        to_del = [u for u, o in self.scene_objects.items() if o.data_key.startswith(prefix)]
        return [self.scene_objects.pop(u) for u in to_del]

    def set_object_visible_by_data_key(self, data_key, visible):
        for o in self.scene_objects.values():
            if o.data_key == data_key:
                o.visible = visible

    def clear_streamlines(self):
        self.streamline_seeds = None
        self.streamline_cache.clear()
        self.streamline_active = False
        self.remove_object_by_data_key("streamlines_live")

    def clear_plane_streamlines(self):
        self.plane_streamline_cache.clear()
        self.plane_streamline_active = False
        self.plane_streamline_plane_idx = -1
        self.remove_object_by_data_key("plane_streamlines_live")

    def reset_all(self):
        for attr, default in [
            ("paths", PathsState()),
            ("pipeline", PipelineFlags()),
            ("preprocess_params", PreprocessParams()),
            ("skeleton_params", SkeletonParams()),
            ("plane_gen_params", PlaneGenerationParams()),
            ("streamline_params", StreamlineParams()),
            ("derived_params", DerivedMetricsParams()),
        ]:
            setattr(self, attr, default)
        self.resolution = np.array([1., 1., 1.])
        self.origin = np.array([0., 0., 0.])
        self.venc = np.array([1., 1., 1.])
        self.rr = 1000.0
        for attr in ["segmask_raw", "segmask_labels", "segmask_binary", "segmask_3d",
                      "skeleton_points", "skeleton_mask", "branch_labels", "flow_raw",
                      "streamline_seeds", "mag_raw"]:
            setattr(self, attr, None)
        self.graph = GraphData()
        self.centerline_paths = []
        self.centerline_node_paths = []
        self.centerline_paths_smooth = []
        self.path_info = []
        self.forks = []
        self.planes = []
        self.streamline_cache = {}
        self.streamline_active = False
        self.plane_streamline_cache = {}
        self.plane_streamline_active = False
        self.plane_streamline_plane_idx = -1
        self.derived = DerivedResults()
        self.scene_objects = {}
        self.current_t = 0
        self.data_loaded = False
        self.ortho_cursor = np.array([0, 0, 0], dtype=int)
        self.selected_path_index = -1

    def snapshot_dict(self):
        def arr(v):
            return None if v is None else np.asarray(v).tolist()
        return {
            "paths": {"segmask_path": self.paths.segmask_path, "flow_path": self.paths.flow_path,
                      "workspace_path": self.paths.workspace_path, "output_dir": self.paths.output_dir},
            "pipeline": {"completed": dict(self.pipeline.completed), "skipped": dict(self.pipeline.skipped)},
            "preprocess_params": self.preprocess_params.to_dict(),
            "skeleton_params": self.skeleton_params.to_dict(),
            "plane_gen_params": self.plane_gen_params.to_dict(),
            "streamline_params": self.streamline_params.to_dict(),
            "derived_params": self.derived_params.to_dict(),
            "resolution": arr(self.resolution),
            "origin": arr(self.origin),
            "venc": arr(self.venc),
            "rr": float(self.rr),
            "segmask_raw": arr(self.segmask_raw),
            "segmask_labels": arr(self.segmask_labels),
            "segmask_binary": arr(self.segmask_binary),
            "segmask_3d": arr(self.segmask_3d),
            "mag_raw": arr(self.mag_raw),
            "skeleton_points": arr(self.skeleton_points),
            "skeleton_mask": arr(self.skeleton_mask),
            "graph": {"points": arr(self.graph.points), "edges": arr(self.graph.edges)},
            "branch_labels": arr(self.branch_labels),
            "centerline_paths": [arr(x) for x in self.centerline_paths],
            "centerline_node_paths": [list(map(int, x)) for x in self.centerline_node_paths],
            "centerline_paths_smooth": [arr(x) for x in self.centerline_paths_smooth],
            "path_info": copy.deepcopy(self.path_info),
            "forks": copy.deepcopy(self.forks),
            "planes": [{"center": arr(p.center), "normal": arr(p.normal), "label": int(p.label),
                        "path_index": int(p.path_index), "distance": float(p.distance),
                        "metrics": copy.deepcopy(p.metrics)} for p in self.planes],
            "flow_raw": arr(self.flow_raw),
            "streamline_seeds": arr(self.streamline_seeds),
            "streamline_active": self.streamline_active,
            "scene_objects": [
                {"uid": o.uid, "name": o.name, "kind": o.kind.value, "data_key": o.data_key,
                 "visible": o.visible, "opacity": o.opacity, "color": o.color,
                 "scalars": o.scalars, "cmap": o.cmap,
                 "clim": list(o.clim) if o.clim else None,
                 "point_size": o.point_size, "line_width": o.line_width,
                 "tube_radius": o.tube_radius, "show_scalar_bar": o.show_scalar_bar,
                 "scalar_bar_title": o.scalar_bar_title, "dynamic": o.dynamic}
                for o in self.scene_objects.values()],
            "current_t": self.current_t, "data_loaded": self.data_loaded,
            "selected_path_index": int(self.selected_path_index),
        }

    def restore_dict(self, d):
        self.paths = PathsState(**{k: d.get("paths", {}).get(k, "") for k in ["segmask_path", "flow_path", "workspace_path", "output_dir"]})
        self.pipeline = PipelineFlags(completed=dict(d.get("pipeline", {}).get("completed", {})),
                                      skipped=dict(d.get("pipeline", {}).get("skipped", {})))
        self.preprocess_params = PreprocessParams.from_dict(d.get("preprocess_params", {}))
        self.skeleton_params = SkeletonParams.from_dict(d.get("skeleton_params", {}))
        self.plane_gen_params = PlaneGenerationParams.from_dict(d.get("plane_gen_params", {}))
        self.streamline_params = StreamlineParams.from_dict(d.get("streamline_params", {}))
        self.derived_params = DerivedMetricsParams.from_dict(d.get("derived_params", {}))
        self.resolution = np.asarray(d.get("resolution", [1, 1, 1]), dtype=float)
        self.origin = np.array([0.0, 0.0, 0.0], dtype=float)
        self.venc = np.asarray(d.get("venc", [1, 1, 1]), dtype=float)
        self.rr = float(d.get("rr", 1000.0))

        def nparr(k, dt=np.float64):
            v = d.get(k)
            return None if v is None else np.asarray(v, dtype=dt)

        self.segmask_raw = nparr("segmask_raw", np.int16)
        self.segmask_labels = nparr("segmask_labels", np.int16)
        self.segmask_binary = None if d.get("segmask_binary") is None else np.asarray(d["segmask_binary"], dtype=bool)
        self.segmask_3d = None if d.get("segmask_3d") is None else np.asarray(d["segmask_3d"], dtype=bool)
        self.mag_raw = nparr("mag_raw")
        self.skeleton_points = nparr("skeleton_points")
        self.skeleton_mask = nparr("skeleton_mask")
        gd = d.get("graph", {})
        self.graph = GraphData(
            points=np.asarray(gd.get("points", []), dtype=float).reshape(-1, 3) if gd.get("points") else np.empty((0, 3)),
            edges=np.asarray(gd.get("edges", []), dtype=int).reshape(-1, 2) if gd.get("edges") else np.empty((0, 2), dtype=int))
        self.branch_labels = nparr("branch_labels")
        self.centerline_paths = [np.asarray(x, dtype=float) for x in d.get("centerline_paths", [])]
        self.centerline_node_paths = [list(map(int, x)) for x in d.get("centerline_node_paths", [])]
        self.centerline_paths_smooth = [np.asarray(x, dtype=float) for x in d.get("centerline_paths_smooth", [])]
        self.path_info = copy.deepcopy(d.get("path_info", []))
        self.forks = copy.deepcopy(d.get("forks", []))
        self.planes = []
        for p in d.get("planes", []):
            self.planes.append(PlaneData(
                center=np.asarray(p["center"], dtype=float), normal=np.asarray(p["normal"], dtype=float),
                label=int(p.get("label", 1)), path_index=int(p.get("path_index", 0)),
                distance=float(p.get("distance", 0.0)), metrics=copy.deepcopy(p.get("metrics", {}))))
        self.flow_raw = nparr("flow_raw")
        self.streamline_seeds = nparr("streamline_seeds")
        self.streamline_cache = {}
        self.streamline_active = bool(d.get("streamline_active", False))
        self.plane_streamline_cache = {}
        self.plane_streamline_active = False
        self.plane_streamline_plane_idx = -1
        self.scene_objects = {}
        for it in d.get("scene_objects", []):
            uid = it["uid"]
            self.scene_objects[uid] = SceneObject(
                uid=uid, name=it["name"], kind=ObjectKind(it["kind"]), data_key=it["data_key"],
                visible=bool(it.get("visible", True)), opacity=float(it.get("opacity", 1.0)),
                color=it.get("color", "white"), scalars=it.get("scalars"),
                cmap=it.get("cmap", "turbo"),
                clim=tuple(it["clim"]) if it.get("clim") else None,
                point_size=int(it.get("point_size", 8)), line_width=int(it.get("line_width", 2)),
                tube_radius=float(it.get("tube_radius", 0.0)),
                show_scalar_bar=bool(it.get("show_scalar_bar", False)),
                scalar_bar_title=it.get("scalar_bar_title"), dynamic=bool(it.get("dynamic", False)))
        self.current_t = int(d.get("current_t", 0))
        self.data_loaded = bool(d.get("data_loaded", False))
        self.selected_path_index = int(d.get("selected_path_index", -1))
