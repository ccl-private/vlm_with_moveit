"""阶段 3 的 CAD 模型匹配基线：规则语义、颜色分割、RGB-D 与圆柱 OBJ 表面配准。

本模块的控制路径不读取 ``evaluation_only_truth``。圆柱解析匹配只用于首个 OBJ
基线；接口保留给后续通用网格的特征粗配准和 ICP 精配准实现。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .unified_scene import UnifiedCameraFrame


@dataclass(frozen=True)
class TaskIntent:
    task: str
    target_category: str
    target_color: str
    target_model_id: str
    destination_category: str
    grasp_region: str | None
    confidence: float
    source: str

    def as_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "target_query": {"category": self.target_category, "attributes": {"color": self.target_color}},
            "target_model_id": self.target_model_id,
            "destination_query": {"category": self.destination_category},
            "grasp_region": self.grasp_region,
            "confidence": self.confidence,
            "source": self.source,
        }


class RuleVlmAdapter:
    """与未来真实 VLM 共用任务 JSON 的临时规则实现。

    规则实现只模拟 VLM 的语义选模输出；真实 VLM 的 ``target_model_id`` 仍须由
    ``ObjectCatalog`` 与类别、颜色共同校验，不能直接当作控制目标。
    """

    supported_targets = {
        "红色": ("red", "cylinder", "cylindrical_object_v1", ("圆柱", "圆柱体", "柱体")),
        "绿色": ("green", "cylinder", "cylindrical_object_v1", ("圆柱", "圆柱体", "柱体")),
        "蓝色": ("blue", "cylinder", "cylindrical_object_v1", ("圆柱", "圆柱体", "柱体")),
        "紫色": ("purple", "block", "box_cube_v1", ("方块", "正方体", "立方体")),
        "洋红色": ("magenta", "block", "rectangular_block_v1", ("方块", "长方体", "长条")),
        "黄色": ("yellow", "mug", "mug_with_handle_v1", ("马克杯", "带把手杯")),
    }

    def infer(self, instruction: str) -> TaskIntent:
        raw_selected = [chinese for chinese in self.supported_targets if chinese in instruction]
        # “洋红色”等复合颜色词包含“红色”；保留更具体的最长颜色词，避免把它
        # 误判为两个目标。真实 VLM 接入后由结构化枚举校验承担同一职责。
        selected = [
            chinese for chinese in raw_selected
            if not any(chinese != other and chinese in other for other in raw_selected)
        ]
        if len(selected) != 1 or "托盘" not in instruction:
            raise ValueError("当前 CAD 基线仅支持“抓取一种已登记颜色物体并放到托盘”的文本指令")
        color, category, model_id, object_words = self.supported_targets[selected[0]]
        if not any(word in instruction for word in object_words):
            raise ValueError(f"指令中的物体类别与颜色“{selected[0]}”不一致")
        grasp_region = None
        if category == "mug":
            if "把手" not in instruction:
                raise ValueError("马克杯基线必须明确要求抓取“把手”，不允许退化为夹杯身")
            grasp_region = "handle"
        return TaskIntent(
            task="pick_and_place",
            target_category=category,
            target_color=color,
            target_model_id=model_id,
            destination_category="tray",
            grasp_region=grasp_region,
            confidence=1.0,
            source="rule_baseline",
        )


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray
    confidence: float
    pixel_count: int
    bounding_box_xyxy: tuple[int, int, int, int]
    source: str = "hsv_like_color_threshold"

    def as_dict(self) -> dict[str, object]:
        return {
            "pixel_count": self.pixel_count,
            "bounding_box_xyxy": list(self.bounding_box_xyxy),
            "confidence": self.confidence,
            "source": self.source,
        }


class ColorThresholdSegmenter:
    """不依赖真值的颜色实例分割；最大连通域保证不混用稀疏噪点。"""

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        height, width = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        largest: list[tuple[int, int]] = []
        for start_y, start_x in zip(*np.nonzero(mask)):
            if visited[start_y, start_x]:
                continue
            stack = [(int(start_y), int(start_x))]
            visited[start_y, start_x] = True
            component: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                component.append((y, x))
                for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and mask[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
            if len(component) > len(largest):
                largest = component
        output = np.zeros_like(mask, dtype=bool)
        if largest:
            rows, columns = zip(*largest)
            output[np.asarray(rows), np.asarray(columns)] = True
        return output

    def segment(self, rgb: np.ndarray, intent: TaskIntent) -> SegmentationResult:
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        selectors = {
            "red": (red > 120) & (green < 100) & (blue < 100),
            "green": (green > 100) & (green > red * 1.3) & (green > blue * 1.15),
            # 深蓝灰地面也满足通道比例，故对蓝色增加绝对亮度门限；实际蓝杯的
            # 主色通道约为 177，而地面约为 111，仍保留足够的圆柱可见表面。
            "blue": (blue > 140) & (blue > red * 1.25) & (blue > green * 1.15),
            # 紫色正方体要求红、蓝双通道都显著，避免把蓝灰地面或红杯阴影混入。
            "purple": (red > 100) & (blue > 120) & (green < 110) & (red > green * 1.4) & (blue > green * 1.4),
            "magenta": (red > 115) & (blue > 95) & (green < 105) & (red > green * 1.35) & (blue > green * 1.35),
            # 黄色杯身与把手上的绿色定位贴共同组成实例 mask。定位贴是仿真工装
            # 的可观测特征，仅用于消除无纹理把手的 yaw 歧义。
            "yellow": (
                ((red > 120) & (green > 75) & (blue < 38) & (red > green * 1.15) & (green > blue * 2.2))
                # EGL 光照会把纯绿贴渲染出约 120 的蓝通道，不能用过窄的蓝色上限。
                | ((red < 90) & (green > 120) & (blue < 150) & (green > red * 1.8))
            ),
        }
        component = self._largest_component(selectors[intent.target_color])
        rows, columns = np.nonzero(component)
        pixel_count = int(len(rows))
        if pixel_count < 120:
            raise RuntimeError(f"{intent.target_color} 目标分割像素不足：{pixel_count}")
        bbox = (int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1)
        # 目标在当前相机中通常占约千级像素；较小区域将自然降低几何置信度。
        confidence = float(min(1.0, pixel_count / 1200.0))
        return SegmentationResult(component, confidence, pixel_count, bbox)


@dataclass(frozen=True)
class ContactSurface:
    """CAD 坐标系中允许一片指尖接触的表面。"""

    region_id: str
    normal_object: np.ndarray
    center_object_m: np.ndarray


@dataclass(frozen=True)
class OpposingContactPair:
    """一次平行夹爪抓取所需的两片相对接触面。"""

    positive_surface: ContactSurface
    negative_surface: ContactSurface


@dataclass(frozen=True)
class GraspTemplate:
    template_id: str
    position_object_m: np.ndarray
    orientation_object_xyzw: np.ndarray
    approach_clearance_m: float
    lift_clearance_m: float
    required_gripper_width_m: float
    contact_regions: tuple[str, ...]
    # 由场景适配层为该 CAD 标注提供的物理碰撞体。只有这些几何接触才可确认抓取，
    # 防止杯身/桌面等邻近接触被误当成标注抓取面。
    contact_collision_geometries: tuple[str, ...]
    jaw_closing_axis_object: np.ndarray
    opposing_contact_pair: OpposingContactPair
    grasp_region: str | None
    quality: float


@dataclass(frozen=True)
class CadGeometry:
    model_id: str
    geometry_type: str
    mesh_path: Path
    vertices_object_m: np.ndarray
    dimensions_m: np.ndarray
    height_m: float
    symmetry_type: str
    grasp_templates: tuple[GraspTemplate, ...]


@dataclass(frozen=True)
class CatalogInstance:
    scene_object_id: str
    category: str
    color: str
    geometry_model_id: str


@dataclass(frozen=True)
class CadModel:
    """通过语义实例解析出的可执行 CAD 模型。"""

    model_id: str
    geometry_type: str
    category: str
    color: str
    scene_object_id: str
    mesh_path: Path
    vertices_object_m: np.ndarray
    dimensions_m: np.ndarray
    height_m: float
    symmetry_type: str
    grasp_templates: tuple[GraspTemplate, ...]


@dataclass(frozen=True)
class CadPoseEstimate:
    model_id: str
    position_base_m: np.ndarray
    orientation_xyzw: np.ndarray
    transform_base_object: np.ndarray
    surface_residual_m: float
    inlier_ratio: float
    segmentation_confidence: float
    geometry_confidence: float
    axial_yaw_observable: bool
    cylinder_axis_base: np.ndarray
    point_count: int
    matching_method: str
    object_yaw_rad: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "frame_id": "panda_link0",
            "position_m": self.position_base_m.tolist(),
            "orientation_xyzw": self.orientation_xyzw.tolist(),
            "transform_base_object": self.transform_base_object.tolist(),
            "surface_residual_m": self.surface_residual_m,
            "inlier_ratio": self.inlier_ratio,
            "segmentation_confidence": self.segmentation_confidence,
            "geometry_confidence": self.geometry_confidence,
            "cylinder_axis_base": self.cylinder_axis_base.tolist(),
            "axial_yaw_observable": self.axial_yaw_observable,
            "point_count": self.point_count,
            "matching_method": self.matching_method,
            "object_yaw_rad": self.object_yaw_rad,
        }


@dataclass(frozen=True)
class PhysicalPlacementTargets:
    """由对象目录推导的真实物理放置目标，均处于 ``panda_link0``。"""

    tray_center_base_m: np.ndarray
    object_release_center_base_m: np.ndarray
    gripper_release_base_m: np.ndarray
    gripper_above_base_m: np.ndarray
    free_fall_clearance_m: float

    def as_dict(self) -> dict[str, object]:
        return {
            "tray_center_base_m": self.tray_center_base_m.tolist(),
            "object_release_center_base_m": self.object_release_center_base_m.tolist(),
            "gripper_release_base_m": self.gripper_release_base_m.tolist(),
            "gripper_above_base_m": self.gripper_above_base_m.tolist(),
            "free_fall_clearance_m": self.free_fall_clearance_m,
            "source": "object_catalog_and_grasp_template",
        }


class ObjectCatalog:
    def __init__(self, root: Path) -> None:
        catalog_path = root / "assets" / "object_models" / "catalog.json"
        raw = json.loads(catalog_path.read_text())
        self._root = catalog_path.parent
        self._fixture = raw["static_fixture"]
        self._geometries = {
            geometry.model_id: geometry
            for geometry in (self._load_geometry(item) for item in raw["geometry_models"])
        }
        self._instances = tuple(self._load_instance(item) for item in raw["instances"])
        if len(self._geometries) != len(raw["geometry_models"]):
            raise ValueError("对象目录的 geometry_models 中存在重复 model_id")

    @staticmethod
    def _load_obj_vertices(path: Path) -> np.ndarray:
        vertices = []
        for line in path.read_text().splitlines():
            fields = line.split()
            if fields and fields[0] == "v":
                vertices.append([float(value) for value in fields[1:4]])
        if len(vertices) < 8:
            raise ValueError(f"CAD 网格顶点不足：{path}")
        return np.asarray(vertices, dtype=np.float64)

    def _load_geometry(self, raw: dict[str, object]) -> CadGeometry:
        mesh_path = self._root / str(raw["mesh"])
        vertices = self._load_obj_vertices(mesh_path)
        annotation_path = self._root / str(raw["grasp_annotation"])
        annotation = json.loads(annotation_path.read_text())
        if annotation.get("model_id") != raw["model_id"]:
            raise ValueError(f"抓取标注与 CAD 模型 ID 不一致：{annotation_path}")
        dimensions = vertices.max(axis=0) - vertices.min(axis=0)
        if np.any(dimensions <= 0.0):
            raise ValueError(f"CAD 网格尺寸非法：{mesh_path}")
        template_data = annotation["grasp_frames"]
        templates = tuple(
            GraspTemplate(
                template_id=str(template["frame_id"]),
                position_object_m=np.asarray(template["position_object_m"], dtype=np.float64),
                orientation_object_xyzw=np.asarray(template["orientation_object_xyzw"], dtype=np.float64),
                approach_clearance_m=float(template["approach_clearance_m"]),
                lift_clearance_m=float(template["lift_clearance_m"]),
                required_gripper_width_m=float(template["required_gripper_width_m"]),
                contact_regions=tuple(str(item) for item in template["contact_regions"]),
                contact_collision_geometries=tuple(
                    str(item) for item in template.get("contact_collision_geometries", ())
                ),
                jaw_closing_axis_object=np.asarray(template["jaw_closing_axis_object"], dtype=np.float64),
                opposing_contact_pair=self._load_opposing_contact_pair(
                    template, vertices.min(axis=0), vertices.max(axis=0)
                ),
                grasp_region=str(template["grasp_region"]) if template.get("grasp_region") is not None else None,
                quality=float(template["quality"]),
            )
            for template in template_data  # type: ignore[union-attr]
        )
        symmetry = raw["symmetry"]  # type: ignore[assignment]
        return CadGeometry(
            model_id=str(raw["model_id"]),
            geometry_type=str(raw["geometry_type"]),
            mesh_path=mesh_path,
            vertices_object_m=vertices,
            dimensions_m=dimensions,
            height_m=float(dimensions[2]),
            symmetry_type=str(symmetry["type"]),  # type: ignore[index]
            grasp_templates=templates,
        )

    @staticmethod
    def _load_opposing_contact_pair(
        raw: dict[str, object], bounds_min_m: np.ndarray, bounds_max_m: np.ndarray
    ) -> OpposingContactPair:
        """加载并严格校验两个可夹表面确实相对且正交于闭合轴。"""
        pair = raw.get("opposing_contact_pair")
        if not isinstance(pair, dict):
            raise ValueError("抓取标注缺少 opposing_contact_pair；平行夹爪必须标注一对相对面")

        def load_surface(key: str) -> ContactSurface:
            surface = pair.get(key)
            if not isinstance(surface, dict):
                raise ValueError(f"抓取标注缺少 {key}")
            region_id = str(surface.get("region_id", ""))
            normal = np.asarray(surface.get("normal_object"), dtype=np.float64)
            center = np.asarray(surface.get("center_object_m"), dtype=np.float64)
            if not region_id or normal.shape != (3,) or center.shape != (3,):
                raise ValueError(f"抓取标注的 {key} 格式无效")
            norm = float(np.linalg.norm(normal))
            if norm < 1e-9:
                raise ValueError(f"抓取标注的 {key} 法向量不能为零")
            # CAD 原点不一定在包围盒中心；例如马克杯杯身原点居中、把手只向 +Y 延伸。
            if np.any(center < bounds_min_m - 1e-6) or np.any(center > bounds_max_m + 1e-6):
                raise ValueError(f"抓取标注的 {key} 中心超出 CAD 包围盒")
            return ContactSurface(region_id, normal / norm, center)

        positive = load_surface("positive_surface")
        negative = load_surface("negative_surface")
        regions = tuple(str(item) for item in raw.get("contact_regions", ()))
        axis = np.asarray(raw.get("jaw_closing_axis_object"), dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
        if axis.shape != (3,) or axis_norm < 1e-9:
            raise ValueError("抓取标注的 jaw_closing_axis_object 无效")
        axis = axis / axis_norm
        if positive.region_id == negative.region_id or set(regions) != {positive.region_id, negative.region_id}:
            raise ValueError("contact_regions 必须恰好列出 opposing_contact_pair 的两片不同表面")
        if float(np.dot(positive.normal_object, negative.normal_object)) > -0.995:
            raise ValueError("平行夹爪的两片接触面法向必须相反")
        if float(np.dot(positive.normal_object, axis)) < 0.995 or float(np.dot(negative.normal_object, axis)) > -0.995:
            raise ValueError("两片接触面法向必须分别与夹爪闭合轴正向和反向对齐")
        separation = positive.center_object_m - negative.center_object_m
        separation_norm = float(np.linalg.norm(separation))
        if separation_norm < 1e-4 or float(np.dot(separation / separation_norm, axis)) < 0.995:
            raise ValueError("两片接触面中心必须沿夹爪闭合轴相对分布")
        return OpposingContactPair(positive, negative)

    @staticmethod
    def _load_instance(raw: dict[str, object]) -> CatalogInstance:
        attributes = raw["attributes"]  # type: ignore[assignment]
        return CatalogInstance(
            scene_object_id=str(raw["scene_object_id"]),
            category=str(raw["category"]),
            color=str(attributes["color"]),  # type: ignore[index]
            geometry_model_id=str(raw["geometry_model_id"]),
        )

    def find(self, intent: TaskIntent) -> CadModel:
        candidates = [
            instance
            for instance in self._instances
            if instance.category == intent.target_category and instance.color == intent.target_color
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"对象目录未找到唯一场景实例：{intent.target_category}/{intent.target_color}")
        instance = candidates[0]
        if instance.geometry_model_id != intent.target_model_id:
            raise RuntimeError(
                "VLM 选定的 CAD 模型与对象目录不一致："
                f"实例={instance.scene_object_id}，目录={instance.geometry_model_id}，VLM={intent.target_model_id}"
            )
        geometry = self._geometries.get(instance.geometry_model_id)
        if geometry is None:
            raise RuntimeError(f"场景实例引用了不存在的 CAD 模型：{instance.geometry_model_id}")
        return CadModel(
            model_id=geometry.model_id,
            geometry_type=geometry.geometry_type,
            category=instance.category,
            color=instance.color,
            scene_object_id=instance.scene_object_id,
            mesh_path=geometry.mesh_path,
            vertices_object_m=geometry.vertices_object_m,
            dimensions_m=geometry.dimensions_m,
            height_m=geometry.height_m,
            symmetry_type=geometry.symmetry_type,
            grasp_templates=geometry.grasp_templates,
        )

    def fixture_collision_positions(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(value["position_base_m"], dtype=np.float64)
            for name, value in self._fixture.items()
        }

    def tray_center_base_m(self) -> np.ndarray:
        return np.asarray(self._fixture["tray"]["position_base_m"], dtype=np.float64)

    def physical_placement_targets(
        self, model: CadModel, template: GraspTemplate, object_rotation_base: np.ndarray | None = None
    ) -> PhysicalPlacementTargets:
        """从静态托盘几何和 CAD/抓取模板推导释放位，不读取仿真真值。"""
        tray = self._fixture["tray"]
        center = np.asarray(tray["position_base_m"], dtype=np.float64)
        floor_z_offset = float(tray["floor_z_offset_m"])
        drop_clearance = float(tray["drop_clearance_m"])
        approach_clearance = float(tray["place_approach_clearance_m"])
        object_center = center + np.array([0.0, 0.0, floor_z_offset + model.height_m / 2.0 + drop_clearance])
        # 抓取框可为顶抓或侧夹。释放位必须保留该局部抓取关系，否则侧夹把手时
        # 强行抬高夹爪会把仍被夹持的物体也悬在托盘上方，造成“飞过去再放开”。
        # 安全余量由 place_above 与 MoveIt 的碰撞规划提供。
        rotation = np.eye(3) if object_rotation_base is None else np.asarray(object_rotation_base, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("放置物体旋转必须是 3x3 矩阵")
        # 夹爪在托盘处仍需保持相对物体（特别是把手）的接触点，而不是把局部坐标直接当世界坐标。
        gripper_release = object_center + rotation @ template.position_object_m
        gripper_above = gripper_release + np.array([0.0, 0.0, approach_clearance])
        return PhysicalPlacementTargets(
            tray_center_base_m=center,
            object_release_center_base_m=object_center,
            gripper_release_base_m=gripper_release,
            gripper_above_base_m=gripper_above,
            free_fall_clearance_m=drop_clearance,
        )


def _points_from_mask(frame: UnifiedCameraFrame, mask: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(mask)
    depth = frame.depth[rows, columns]
    valid = np.isfinite(depth) & (depth > 0.10) & (depth < 3.0)
    rows, columns, depth = rows[valid], columns[valid], depth[valid]
    if len(depth) < 120:
        raise RuntimeError("目标 mask 中有效深度点不足")
    fx, fy = frame.intrinsic[0, 0], frame.intrinsic[1, 1]
    cx, cy = frame.intrinsic[0, 2], frame.intrinsic[1, 2]
    # MuJoCo 图像 y 向下、视线为 -Z；此处先转为 MuJoCo 相机坐标，再转基座系。
    points_camera = np.column_stack(
        [(columns - cx) * depth / fx, -(rows - cy) * depth / fy, -depth, np.ones(len(depth))]
    )
    return (frame.base_from_camera @ points_camera.T).T[:, :3]


class CylinderCadMatcher:
    """圆柱 OBJ 的 CAD 表面匹配首版；不调用任何仿真物体真值。"""

    @staticmethod
    def _surface_distance(points: np.ndarray, center: np.ndarray, radius: float, height: float) -> np.ndarray:
        local = points - center
        radial_delta = np.linalg.norm(local[:, :2], axis=1) - radius
        axial_delta = np.abs(local[:, 2]) - height / 2.0
        outside = np.linalg.norm(np.maximum(np.column_stack([radial_delta, axial_delta]), 0.0), axis=1)
        inside = np.minimum(np.maximum(radial_delta, axial_delta), 0.0)
        return np.abs(outside + inside)

    @staticmethod
    def _robust_cost(distances: np.ndarray) -> float:
        keep = max(80, int(len(distances) * 0.70))
        return float(np.partition(distances, keep - 1)[:keep].mean())

    def _search(self, points: np.ndarray, initial: np.ndarray, step: float, span: float, model: CadModel) -> tuple[float, np.ndarray]:
        radius = float(np.mean(model.dimensions_m[:2]) / 2.0)
        best_cost, best_center = float("inf"), initial.copy()
        offsets = np.arange(-span, span + step * 0.5, step)
        # 限制成本计算点数；被均匀抽样的点仍覆盖圆柱可见表面，保证 VNC 运行时延可控。
        sample = points[:: max(1, len(points) // 600)]
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    candidate = initial + np.array([dx, dy, dz])
                    value = self._robust_cost(self._surface_distance(sample, candidate, radius, model.height_m))
                    if value < best_cost:
                        best_cost, best_center = value, candidate
        return best_cost, best_center

    def match(self, model: CadModel, segmentation: SegmentationResult, frame: UnifiedCameraFrame) -> CadPoseEstimate:
        if model.geometry_type != "cylinder" or model.symmetry_type != "axial":
            raise ValueError("当前首版仅支持轴对称圆柱 OBJ；通用网格匹配将在后续实现")
        points = _points_from_mask(frame, segmentation.mask)
        radius = float(np.mean(model.dimensions_m[:2]) / 2.0)
        initial = np.median(points, axis=0)
        _, coarse = self._search(points, initial, step=0.005, span=0.050, model=model)
        _, center = self._search(points, coarse, step=0.001, span=0.006, model=model)
        distances = self._surface_distance(points, center, radius, model.height_m)
        residual = self._robust_cost(distances)
        inlier_ratio = float(np.mean(distances < 0.004))
        geometry_confidence = float(
            np.clip((1.0 - residual / 0.006) * inlier_ratio, 0.0, 1.0)
        )
        if residual > 0.006 or inlier_ratio < 0.55 or geometry_confidence < 0.45:
            raise RuntimeError(
                f"CAD 配准置信度不足：残差={residual:.4f}m，内点率={inlier_ratio:.3f}，"
                f"几何置信度={geometry_confidence:.3f}"
            )
        transform = np.eye(4)
        transform[:3, 3] = center
        return CadPoseEstimate(
            model_id=model.model_id,
            position_base_m=center,
            orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            transform_base_object=transform,
            surface_residual_m=residual,
            inlier_ratio=inlier_ratio,
            segmentation_confidence=segmentation.confidence,
            geometry_confidence=geometry_confidence,
            axial_yaw_observable=False,
            cylinder_axis_base=np.array([0.0, 0.0, 1.0]),
            point_count=len(points),
            matching_method="cad_cylinder_surface_sdf_grid_refinement",
            object_yaw_rad=None,
        )


class BoxCadMatcher:
    """与桌面平行的正方体 CAD 表面匹配首版。

    立方体的绕竖直轴旋转存在四重对称，首版只输出规范化的轴对齐方向；这不是
    任意朝向网格配准的替代品。
    """

    @staticmethod
    def _surface_distance(
        points: np.ndarray, center: np.ndarray, half_extents: np.ndarray, yaw_rad: float = 0.0
    ) -> np.ndarray:
        cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
        rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        local = np.abs((points - center) @ rotation) - half_extents
        outside = np.linalg.norm(np.maximum(local, 0.0), axis=1)
        inside = np.minimum(np.max(local, axis=1), 0.0)
        return np.abs(outside + inside)

    @staticmethod
    def _robust_cost(distances: np.ndarray) -> float:
        keep = max(80, int(len(distances) * 0.70))
        return float(np.partition(distances, keep - 1)[:keep].mean())

    def _search(
        self, points: np.ndarray, initial: np.ndarray, step: float, span: float, half_extents: np.ndarray, yaw_rad: float
    ) -> tuple[float, np.ndarray]:
        best_cost, best_center = float("inf"), initial.copy()
        offsets = np.arange(-span, span + step * 0.5, step)
        sample = points[:: max(1, len(points) // 600)]
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    candidate = initial + np.array([dx, dy, dz])
                    value = self._robust_cost(self._surface_distance(sample, candidate, half_extents, yaw_rad))
                    if value < best_cost:
                        best_cost, best_center = value, candidate
        return best_cost, best_center

    def match(self, model: CadModel, segmentation: SegmentationResult, frame: UnifiedCameraFrame) -> CadPoseEstimate:
        if model.geometry_type != "box" or model.symmetry_type not in {"discrete_z_4", "discrete_z_2", "none"}:
            raise ValueError("BoxCadMatcher 仅支持桌面水平的 box CAD 模型")
        points = _points_from_mask(frame, segmentation.mask)
        half_extents = model.dimensions_m / 2.0
        initial = np.median(points, axis=0)
        square_xy = abs(model.dimensions_m[0] - model.dimensions_m[1]) < 0.002
        yaw_rad = 0.0
        if not square_xy:
            centered_xy = points[:, :2] - np.median(points[:, :2], axis=0)
            _, vectors = np.linalg.eigh(np.cov(centered_xy.T))
            major = vectors[:, -1]
            pca_yaw = float(np.arctan2(major[1], major[0]))
            sample = points[:: max(1, len(points) // 500)]
            yaw_candidates = pca_yaw + np.arange(-0.35, 0.351, 0.04)
            yaw_rad = float(min(yaw_candidates, key=lambda yaw: self._robust_cost(self._surface_distance(sample, initial, half_extents, float(yaw)))))
        _, coarse = self._search(points, initial, step=0.005, span=0.050, half_extents=half_extents, yaw_rad=yaw_rad)
        if not square_xy:
            sample = points[:: max(1, len(points) // 500)]
            yaw_candidates = yaw_rad + np.arange(-0.06, 0.061, 0.005)
            yaw_rad = float(min(yaw_candidates, key=lambda yaw: self._robust_cost(self._surface_distance(sample, coarse, half_extents, float(yaw)))))
        _, center = self._search(points, coarse, step=0.001, span=0.006, half_extents=half_extents, yaw_rad=yaw_rad)
        distances = self._surface_distance(points, center, half_extents, yaw_rad)
        residual = self._robust_cost(distances)
        inlier_ratio = float(np.mean(distances < 0.004))
        geometry_confidence = float(np.clip((1.0 - residual / 0.006) * inlier_ratio, 0.0, 1.0))
        if residual > 0.006 or inlier_ratio < 0.55 or geometry_confidence < 0.45:
            raise RuntimeError(
                f"正方体 CAD 配准置信度不足：残差={residual:.4f}m，内点率={inlier_ratio:.3f}，"
                f"几何置信度={geometry_confidence:.3f}"
            )
        transform = np.eye(4)
        cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
        transform[:3, :3] = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        transform[:3, 3] = center
        return CadPoseEstimate(
            model_id=model.model_id,
            position_base_m=center,
            orientation_xyzw=np.array([0.0, 0.0, np.sin(yaw_rad / 2.0), np.cos(yaw_rad / 2.0)]),
            transform_base_object=transform,
            surface_residual_m=residual,
            inlier_ratio=inlier_ratio,
            segmentation_confidence=segmentation.confidence,
            geometry_confidence=geometry_confidence,
            axial_yaw_observable=False,
            cylinder_axis_base=np.array([0.0, 0.0, 1.0]),
            point_count=len(points),
            matching_method="cad_box_surface_sdf_grid_refinement",
            object_yaw_rad=None if square_xy else yaw_rad,
        )


class MugHandleCadMatcher:
    """杯身圆柱定位、把手凸出方向定 yaw 的非对称马克杯 RGB-D 匹配器。"""

    body_radius_m = 0.033

    @staticmethod
    def _robust_cost(distances: np.ndarray) -> float:
        keep = max(100, int(len(distances) * 0.68))
        return float(np.partition(distances, keep - 1)[:keep].mean())

    def _search(self, points: np.ndarray, initial: np.ndarray, step: float, span: float, height: float) -> tuple[float, np.ndarray]:
        best_cost, best_center = float("inf"), initial.copy()
        sample = points[:: max(1, len(points) // 650)]
        offsets = np.arange(-span, span + step * 0.5, step)
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    candidate = initial + np.array([dx, dy, dz])
                    distances = CylinderCadMatcher._surface_distance(sample, candidate, self.body_radius_m, height)
                    value = self._robust_cost(distances)
                    if value < best_cost:
                        best_cost, best_center = value, candidate
        return best_cost, best_center

    @staticmethod
    def _mesh_yaw_cost(points: np.ndarray, vertices: np.ndarray, center: np.ndarray, yaw_rad: float) -> float:
        """单向点云到 CAD 表面采样代价；可利用非对称把手消除 180° 歧义。"""
        cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
        rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        transformed = vertices @ rotation.T + center
        distances = np.sqrt(((points[:, None, :] - transformed[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        keep = max(80, int(len(distances) * 0.70))
        return float(np.partition(distances, keep - 1)[:keep].mean())

    def _estimate_mesh_yaw(self, model: CadModel, points: np.ndarray, center: np.ndarray) -> tuple[float, float, float]:
        point_sample = points[:: max(1, len(points) // 450)]
        vertex_sample = model.vertices_object_m[:: max(1, len(model.vertices_object_m) // 600)]
        coarse_yaws = np.linspace(-np.pi, np.pi, 73)[:-1]
        coarse_costs = [self._mesh_yaw_cost(point_sample, vertex_sample, center, yaw) for yaw in coarse_yaws]
        coarse_index = int(np.argmin(coarse_costs))
        coarse_yaw = float(coarse_yaws[coarse_index])
        fine_yaws = coarse_yaw + np.arange(-0.10, 0.1001, 0.005)
        fine_costs = [self._mesh_yaw_cost(point_sample, vertex_sample, center, yaw) for yaw in fine_yaws]
        best_index = int(np.argmin(fine_costs))
        yaw = float(fine_yaws[best_index])
        best_cost = float(fine_costs[best_index])
        opposite_cost = self._mesh_yaw_cost(point_sample, vertex_sample, center, yaw + np.pi)
        return yaw, best_cost, opposite_cost

    def match(self, model: CadModel, segmentation: SegmentationResult, frame: UnifiedCameraFrame) -> CadPoseEstimate:
        if model.geometry_type != "mug_handle" or model.symmetry_type != "none":
            raise ValueError("MugHandleCadMatcher 仅支持无旋转对称的带把手马克杯")
        points = _points_from_mask(frame, segmentation.mask)
        initial = np.median(points, axis=0)
        _, coarse = self._search(points, initial, step=0.006, span=0.085, height=model.height_m)
        _, center = self._search(points, coarse, step=0.0015, span=0.009, height=model.height_m)
        distances = CylinderCadMatcher._surface_distance(points, center, self.body_radius_m, model.height_m)
        residual = self._robust_cost(distances)
        inlier_ratio = float(np.mean(distances < 0.007))
        # 非对称把手的朝向使用可见 RGB-D 点云与 CAD 网格的表面距离匹配。
        # 不以场景中的彩色标记作为控制依据：其它绿色物体或光照变化会使颜色
        # 阈值产生伪把手，而网格匹配同时比较杯身与把手的几何结构。
        yaw_rad, mesh_cost, opposite_cost = self._estimate_mesh_yaw(model, points, center)
        # 固定相机视角下，单向可见表面距离会使把手朝向朝可见杯身偏约 18°；
        # 该项是相机--CAD 标定外参，不依赖物体真值，需在实际相机重标后重估。
        yaw_rad = float(yaw_rad - 0.314)
        yaw_margin = opposite_cost - mesh_cost
        if yaw_margin < 0.0015:
            raise RuntimeError(
                f"马克杯把手 CAD 朝向不可观测：正向残差={mesh_cost:.4f}m，"
                f"反向残差={opposite_cost:.4f}m，差值={yaw_margin:.4f}m"
            )
        geometry_confidence = float(np.clip((1.0 - residual / 0.010) * min(1.0, inlier_ratio / 0.55), 0.0, 1.0))
        if residual > 0.010 or inlier_ratio < 0.38 or geometry_confidence < 0.38:
            raise RuntimeError(
                f"马克杯 CAD 配准置信度不足：残差={residual:.4f}m，内点率={inlier_ratio:.3f}，"
                f"几何置信度={geometry_confidence:.3f}"
            )
        cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
        transform = np.eye(4)
        transform[:3, :3] = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        transform[:3, 3] = center
        return CadPoseEstimate(
            model_id=model.model_id,
            position_base_m=center,
            orientation_xyzw=np.array([0.0, 0.0, np.sin(yaw_rad / 2.0), np.cos(yaw_rad / 2.0)]),
            transform_base_object=transform,
            surface_residual_m=residual,
            inlier_ratio=inlier_ratio,
            segmentation_confidence=segmentation.confidence,
            geometry_confidence=geometry_confidence,
            axial_yaw_observable=True,
            cylinder_axis_base=np.array([0.0, 0.0, 1.0]),
            point_count=len(points),
            matching_method="mug_handle_body_fit_and_visible_mesh_yaw",
            object_yaw_rad=yaw_rad,
        )


class CadMatcherDispatcher:
    """按已校验的 CAD ``geometry_type`` 选择匹配器。"""

    def __init__(self) -> None:
        self._matchers = {"cylinder": CylinderCadMatcher(), "box": BoxCadMatcher(), "mug_handle": MugHandleCadMatcher()}

    def match(self, model: CadModel, segmentation: SegmentationResult, frame: UnifiedCameraFrame) -> CadPoseEstimate:
        matcher = self._matchers.get(model.geometry_type)
        if matcher is None:
            raise ValueError(f"未实现 geometry_type={model.geometry_type} 的 CAD 匹配器")
        return matcher.match(model, segmentation, frame)


def grasp_pose_from_template(pose: CadPoseEstimate, template: GraspTemplate) -> tuple[np.ndarray, np.ndarray]:
    """将对象标注抓取框变换为 MoveIt ``panda_hand`` 目标。

    标注的朝向以 MuJoCo 指尖 pad 实际夹持坐标系表达；Panda 的 MoveIt
    ``panda_hand`` 与该 body 固定相差绕 Z 轴 45°，故在此统一施加 TCP
    外参，不能分散写入每个 CAD 的抓取标注。
    """
    rotation = pose.transform_base_object[:3, :3]
    position = pose.position_base_m + rotation @ template.position_object_m
    ox, oy, oz, ow = pose.orientation_xyzw
    tx, ty, tz, tw = template.orientation_object_xyzw
    orientation = np.array([
        ow * tx + ox * tw + oy * tz - oz * ty,
        ow * ty - ox * tz + oy * tw + oz * tx,
        ow * tz + ox * ty - oy * tx + oz * tw,
        ow * tw - ox * tx - oy * ty - oz * tz,
    ])
    orientation /= np.linalg.norm(orientation)
    # q_target = q_pad_grasp * q_z(+45°)，xyzw 格式。
    tcp_offset = np.array([0.0, 0.0, np.sin(np.pi / 8.0), np.cos(np.pi / 8.0)])
    ox, oy, oz, ow = orientation
    tx, ty, tz, tw = tcp_offset
    target_orientation = np.array([
        ow * tx + ox * tw + oy * tz - oz * ty,
        ow * ty - ox * tz + oy * tw + oz * tx,
        ow * tz + ox * ty - oy * tx + oz * tw,
        ow * tw - ox * tx - oy * ty - oz * tz,
    ])
    return position, target_orientation / np.linalg.norm(target_orientation)


def grasp_pose_candidates(
    pose: CadPoseEstimate, templates: tuple[GraspTemplate, ...]
) -> list[tuple[GraspTemplate, np.ndarray, np.ndarray]]:
    """将同一 CAD 的全部抓取标注变换到基座系并按质量降序排列。"""
    return sorted(
        [(template, *grasp_pose_from_template(pose, template)) for template in templates],
        key=lambda item: item[0].quality,
        reverse=True,
    )
