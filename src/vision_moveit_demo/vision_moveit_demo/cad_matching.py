"""阶段 3 的 CAD 模型匹配：规则语义、颜色分割与 RGB-D 点云配准。

控制路径不读取 ``evaluation_only_truth``。圆柱、盒体和带把手马克杯保留各自的
高可靠匹配器；``geometry_type=mesh`` 则使用桌面物体的通用 OBJ 网格兜底匹配。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
import open3d as o3d

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
        "橙色": ("orange", "rabbit", "orange_rabbit_ears_v1", ("兔子", "小兔", "兔兔")),
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
        if category == "rabbit":
            if "耳朵" not in instruction:
                raise ValueError("兔子基线必须明确要求抓取“耳朵”，不允许退化为夹身体")
            grasp_region = "ears"
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

    @staticmethod
    def _spatial_roi_mask(
        frame: UnifiedCameraFrame,
        center_base_m: np.ndarray,
        lateral_radius_m: float,
        z_min_m: float,
        z_max_m: float,
    ) -> np.ndarray:
        """以粗 pose 建立基座系深度 ROI，移除与目标颜色相近的桌面/工装。

        这是粗到精配准之间的常规工业步骤：ROI 只依赖前一阶段估计和已标定
        RGB-D，不读取物体真值。上方的 z 下限刻意排除桌面平面，保留杯身与把手。
        """
        if lateral_radius_m <= 0.0 or z_max_m <= z_min_m:
            raise ValueError("空间 ROI 参数无效")
        depth = frame.depth
        rows, columns = np.indices(depth.shape)
        valid = np.isfinite(depth) & (depth > 0.10) & (depth < 3.0)
        fx, fy = frame.intrinsic[0, 0], frame.intrinsic[1, 1]
        cx, cy = frame.intrinsic[0, 2], frame.intrinsic[1, 2]
        homogeneous_camera = np.stack(
            (
                (columns - cx) * depth / fx,
                -(rows - cy) * depth / fy,
                -depth,
                np.ones_like(depth),
            ),
            axis=-1,
        )
        base_points = homogeneous_camera @ frame.base_from_camera.T
        horizontal_delta = base_points[..., :2] - center_base_m[:2]
        return (
            valid
            & (np.linalg.norm(horizontal_delta, axis=-1) <= lateral_radius_m)
            & (base_points[..., 2] >= z_min_m)
            & (base_points[..., 2] <= z_max_m)
        )

    def segment(
        self,
        rgb: np.ndarray,
        intent: TaskIntent,
        *,
        frame: UnifiedCameraFrame | None = None,
        expected_position_base_m: np.ndarray | None = None,
        lateral_radius_m: float | None = None,
        z_min_m: float | None = None,
        z_max_m: float | None = None,
    ) -> SegmentationResult:
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
            # 只取黄色杯子网格（杯身和把手），不把独立的绿色圆柱混入。旧版绿色
            # 定位贴已移除；仅依赖最大连通域会令离相机更近的绿色圆柱被误识别为马克杯。
            "yellow": (red > 120) & (green > 75) & (blue < 38) & (red > green * 1.15) & (red <= green * 1.75) & (green > blue * 2.2),
            # 橙色兔子采用高饱和橙色，刻意避开棕色桌面、红色圆柱和黄色马克杯。
            "orange": (red > 165) & (green > 45) & (green < 125) & (blue < 75) & (red > green * 1.75),
        }
        candidate = selectors[intent.target_color]
        source = "hsv_like_color_threshold"
        roi_arguments = (expected_position_base_m, lateral_radius_m, z_min_m, z_max_m)
        if any(value is not None for value in roi_arguments):
            if frame is None or any(value is None for value in roi_arguments):
                raise ValueError("空间 ROI 分割需要 frame、预估位置、半径和完整 z 范围")
            roi = self._spatial_roi_mask(
                frame,
                np.asarray(expected_position_base_m, dtype=np.float64),
                float(lateral_radius_m),
                float(z_min_m),
                float(z_max_m),
            )
            # 颜色先验对相邻物体仍有价值，但高光会令杯身变白。因此 ROI 内的
            # 高亮低饱和前景也可保留；桌面已由 z 门限排除。
            bright_foreground = (red > 175) & (green > 145) & (blue < 120)
            candidate = roi & (candidate | bright_foreground)
            source = "color_threshold_with_coarse_pose_depth_roi"
        component = self._largest_component(candidate)
        rows, columns = np.nonzero(component)
        pixel_count = int(len(rows))
        if pixel_count < 120:
            raise RuntimeError(f"{intent.target_color} 目标分割像素不足：{pixel_count}")
        bbox = (int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1)
        # 目标在当前相机中通常占约千级像素；较小区域将自然降低几何置信度。
        confidence = float(min(1.0, pixel_count / 1200.0))
        return SegmentationResult(component, confidence, pixel_count, bbox, source=source)


@dataclass(frozen=True)
class ContactSurface:
    """CAD 坐标系中允许一片指尖接触的表面。"""

    region_id: str
    normal_object: np.ndarray
    center_object_m: np.ndarray
    patch_radius_m: float | None = None


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
    triangles_object: np.ndarray
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
    triangles_object: np.ndarray
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
    # 对需要绕 Z 定向的非对称抓取，整体 CAD 残差不足以证明夹持区真的可见。
    # 这两个字段由多视角关键区域匹配器填写；其余 matcher 保持 None。
    critical_region_support: float | None = None
    critical_region_margin: float | None = None
    registration_diagnostics: dict[str, object] | None = None

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
            "critical_region_support": self.critical_region_support,
            "critical_region_margin": self.critical_region_margin,
            "registration_diagnostics": self.registration_diagnostics,
        }


@dataclass(frozen=True)
class GraspRoiEstimate:
    """由抓取接触区得到的受限局部 pose 精修结果。

    ``global_pose`` 只负责把机器人带进观察工作区；本结构中的 ``pose`` 是在
    CAD 抓取标注的两片接触面都获得深度支持后才可用于闭爪的 pose。两者分开
    记录，避免局部把手点云悄悄覆盖整体 CAD 匹配的审计信息。
    """

    pose: CadPoseEstimate
    template_id: str
    positive_support: float
    negative_support: float
    positive_residual_m: float
    negative_residual_m: float
    supporting_view_count: int
    alternative_margin: float
    translation_correction_m: float
    rotation_correction_deg: float
    selected_yaw_offset_rad: float
    # 这些是与最终 pose 共用同一双接触面证据门槛的微小局部朝向。不包含任何
    # 仅因 IK 可达却没有深度支持的候选；调用方可在这一小集合中联合选择可达
    # 的夹爪姿态。
    reachability_validated_yaw_offsets_rad: tuple[float, ...] = (0.0,)

    def as_dict(self) -> dict[str, object]:
        return {
            "template_id": self.template_id,
            "positive_surface_support": self.positive_support,
            "negative_surface_support": self.negative_support,
            "positive_surface_residual_m": self.positive_residual_m,
            "negative_surface_residual_m": self.negative_residual_m,
            "supporting_view_count": self.supporting_view_count,
            "alternative_yaw_margin": self.alternative_margin,
            "translation_correction_m": self.translation_correction_m,
            "rotation_correction_deg": self.rotation_correction_deg,
            "selected_yaw_offset_rad": self.selected_yaw_offset_rad,
            "reachability_validated_yaw_offsets_rad": list(self.reachability_validated_yaw_offsets_rad),
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
    def _load_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
        """读取 OBJ 顶点和三角面；通用兜底不接受没有表面的点云文件。"""
        if path.suffix.lower() != ".obj":
            raise ValueError(f"当前通用网格加载器仅支持 OBJ：{path}")
        vertices = []
        triangles: list[tuple[int, int, int]] = []
        for line in path.read_text().splitlines():
            fields = line.split()
            if fields and fields[0] == "v":
                vertices.append([float(value) for value in fields[1:4]])
            elif fields and fields[0] == "f":
                indices: list[int] = []
                for token in fields[1:]:
                    index_text = token.split("/", 1)[0]
                    index = int(index_text)
                    if index == 0:
                        raise ValueError(f"OBJ 面索引不能为零：{path}")
                    indices.append(index - 1 if index > 0 else len(vertices) + index)
                if len(indices) < 3:
                    raise ValueError(f"OBJ 面顶点不足：{path}")
                triangles.extend((indices[0], indices[index], indices[index + 1]) for index in range(1, len(indices) - 1))
        if len(vertices) < 8:
            raise ValueError(f"CAD 网格顶点不足：{path}")
        vertex_array = np.asarray(vertices, dtype=np.float64)
        triangle_array = np.asarray(triangles, dtype=np.int64)
        if len(triangle_array) == 0 or np.any(triangle_array < 0) or np.any(triangle_array >= len(vertex_array)):
            raise ValueError(f"CAD OBJ 缺少有效三角面：{path}")
        return vertex_array, triangle_array

    def _load_geometry(self, raw: dict[str, object]) -> CadGeometry:
        mesh_path = self._root / str(raw["mesh"])
        vertices, triangles = self._load_obj_mesh(mesh_path)
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
            triangles_object=triangles,
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
            radius = surface.get("patch_radius_m")
            if radius is not None and (not np.isfinite(float(radius)) or float(radius) <= 0.0):
                raise ValueError(f"抓取标注的 {key} patch_radius_m 必须为正数")
            return ContactSurface(region_id, normal / norm, center, float(radius) if radius is not None else None)

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
            triangles_object=geometry.triangles_object,
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

    @staticmethod
    def _yaw_distance(first: float, second: float) -> float:
        return float(abs(np.arctan2(np.sin(first - second), np.cos(first - second))))

    @staticmethod
    def _rotation_z(yaw_rad: float) -> np.ndarray:
        cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
        return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])

    @staticmethod
    def _voxel_average(points: np.ndarray, voxel_m: float = 0.0025, maximum_points: int = 650) -> np.ndarray:
        """在单帧内做小体素平均，降低深度噪声而不混合不同观察方向。

        不能把所有相机帧先体素融合：背面未观测和前景遮挡会被错误平均。这里
        仅对同一已标定 RGB-D 帧的相邻表面点平均，随后每帧相同上限采样。
        """
        if len(points) == 0:
            raise ValueError("体素平均不能处理空点云")
        keys = np.floor(points / voxel_m).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        averaged = np.column_stack(
            [np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)]
        )
        if len(averaged) <= maximum_points:
            return averaged
        return averaged[:: max(1, len(averaged) // maximum_points)]

    @staticmethod
    def _grasp_critical_vertices(model: CadModel) -> np.ndarray:
        """从 CAD 抓取标注派生应被视觉确认的把手外竖条表面。

        这里不写死 ``mug_with_handle_v1`` 的把手坐标。夹持对的两个接触面中点是
        每个 SKU 都必须标注的量；以该中点周围一个随杯高缩放的小包络提取网格
        表面。它既覆盖两指会接触的外竖条，又不会把占多数的圆柱杯身混进来。
        """
        templates = [template for template in model.grasp_templates if template.grasp_region == "handle"]
        if not templates:
            raise RuntimeError("马克杯 CAD 缺少 handle 抓取标注，无法验证关键夹持区")
        template = max(templates, key=lambda item: item.quality)
        pair = template.opposing_contact_pair
        midpoint = (pair.positive_surface.center_object_m + pair.negative_surface.center_object_m) / 2.0
        # 0.36 倍杯高在本 SKU 上为 36 mm：覆盖 56 mm 高的外竖条及其边缘，
        # 同时保持远小于杯身半径到把手距离，避免退化为“只匹配杯身”。
        radius_m = float(np.clip(model.height_m * 0.36, 0.020, 0.045))
        vertices = model.vertices_object_m
        selected = vertices[np.linalg.norm(vertices - midpoint, axis=1) <= radius_m]
        if len(selected) < 24:
            raise RuntimeError("马克杯抓取关键区 CAD 网格采样不足")
        # OBJ 的三角划分密度可能不均匀；均匀抽样上限使分数不受模型导出细节支配。
        return selected[:: max(1, len(selected) // 180)]

    @staticmethod
    def _critical_region_evidence(
        clouds: list[np.ndarray],
        critical_vertices_object: np.ndarray,
        center: np.ndarray,
        yaw_rad: float,
        support_distance_m: float = 0.012,
        view_weights: np.ndarray | None = None,
    ) -> tuple[float, float, int]:
        """返回最佳单视角的关键区覆盖率、近邻残差和支持视角编号。

        不把所有视角先拼成一团：一个相机看不到把手是正常遮挡，不应稀释另一
        相机确实看见把手的证据。CAD→点云方向衡量的是“计划夹持区是否真的有
        观测支持”，正是工业系统里用于抓取前验真的局部 score。
        """
        transformed = critical_vertices_object @ MugHandleCadMatcher._rotation_z(yaw_rad).T + center
        if view_weights is None:
            view_weights = np.ones(len(clouds), dtype=np.float64)
        if len(view_weights) != len(clouds) or np.any(view_weights <= 0.0):
            raise ValueError("关键抓取区的观察视角权重无效")
        best_support, best_residual, best_index = -1.0, float("inf"), -1
        for index, cloud in enumerate(clouds):
            sample = cloud[:: max(1, len(cloud) // 500)]
            squared = ((transformed[:, None, :] - sample[None, :, :]) ** 2).sum(axis=2)
            distances = np.sqrt(squared.min(axis=1))
            # ``view_weights`` 来自主动观察规划对该接触区的预计可见性。它不会
            # 伪造点云，也不决定 pose；只避免一个规划上背对把手的低信息视角因
            # 偶然贴合杯身边缘而压过正对把手的观测。最大权重恒为 1，故分数仍
            # 可解释为关键 CAD 表面得到的有效覆盖率。
            support = float(np.mean(distances <= support_distance_m)) * float(view_weights[index])
            keep = max(8, int(len(distances) * 0.35))
            residual = float(np.partition(distances, keep - 1)[:keep].mean())
            if (support, -residual) > (best_support, -best_residual):
                best_support, best_residual, best_index = support, residual, index
        return best_support, best_residual, best_index

    def _match_points(
        self,
        model: CadModel,
        points: np.ndarray,
        segmentation_confidence: float,
        matching_method: str,
        fixed_camera_yaw_correction_rad: float,
    ) -> CadPoseEstimate:
        if model.geometry_type != "mug_handle" or model.symmetry_type != "none":
            raise ValueError("MugHandleCadMatcher 仅支持无旋转对称的带把手马克杯")
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
        # 这个补偿只属于旧的固定相机单帧路径。腕部多视角点云已经变换到统一
        # 基座系，不能把固定相机的经验偏置带入，否则会把真实的多视角证据旋错。
        yaw_rad = float(yaw_rad + fixed_camera_yaw_correction_rad)
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
            segmentation_confidence=segmentation_confidence,
            geometry_confidence=geometry_confidence,
            axial_yaw_observable=True,
            cylinder_axis_base=np.array([0.0, 0.0, 1.0]),
            point_count=len(points),
            matching_method=matching_method,
            object_yaw_rad=yaw_rad,
        )

    def match(self, model: CadModel, segmentation: SegmentationResult, frame: UnifiedCameraFrame) -> CadPoseEstimate:
        correction = -0.314 if frame.name == "camera_fixed" else 0.0
        return self._match_points(
            model,
            _points_from_mask(frame, segmentation.mask),
            segmentation.confidence,
            "mug_handle_body_fit_and_visible_mesh_yaw",
            correction,
        )

    def match_multiview(
        self,
        model: CadModel,
        observations: list[tuple[SegmentationResult, UnifiedCameraFrame]],
    ) -> CadPoseEstimate:
        """将多个已标定的腕部 RGB-D 帧在基座系联合精配准。

        固定相机只负责发现、粗 ROI 和安全接近；它不进入这个最终抓取 pose 的
        点云融合，也不作为 yaw 先验。每帧独立限采样，避免近距离视角因像素
        更密而淹没其它方向。运行时不读取对象真值。
        """
        if len(observations) < 2:
            raise ValueError("马克杯多视角精配准至少需要两帧 RGB-D 观测")
        if any(frame.name != "camera_wrist" for _, frame, *_ in observations):
            raise ValueError("马克杯最终多视角 pose 只能由腕部 RGB-D 观测生成")
        clouds = []
        confidences = []
        visibility_weights = []
        for observation in observations:
            segmentation, frame = observation[:2]
            cloud = _points_from_mask(frame, segmentation.mask)
            clouds.append(self._voxel_average(cloud))
            confidences.append(segmentation.confidence)
            visibility_weights.append(float(observation[2]) if len(observation) >= 3 else 1.0)
        # 最有希望看到已标注夹持区的视角作为 1.0；其它视角保留 0.15 的贡献。
        # 这允许不同腕部滚转补充遮挡面，同时避免预计背对把手的画面因杯身边缘
        # 的偶然局部贴合主导最终 yaw；最终仍有粗候选 gate 和物理双侧接触。
        raw_weights = np.asarray(visibility_weights, dtype=np.float64)
        if np.any(~np.isfinite(raw_weights)) or np.any(raw_weights < 0.0):
            raise ValueError("主动观察可见性权重非法")
        view_weights = (0.15 + raw_weights) / float(np.max(0.15 + raw_weights))
        all_points = np.concatenate(clouds, axis=0)
        initial = np.median(all_points, axis=0)
        _, coarse = self._search(all_points, initial, step=0.006, span=0.085, height=model.height_m)
        _, body_center = self._search(all_points, coarse, step=0.0015, span=0.009, height=model.height_m)
        critical_vertices = self._grasp_critical_vertices(model)

        # 第一轮仅枚举方位，第二轮才在少数候选周围做平移搜索。这样不会把杯身
        # 的大面积表面作为 yaw 证据，又能修复多视角圆柱拟合在半遮挡下的数毫米
        # 中心偏差。整个流程只使用 RGB-D、外参和 CAD 抓取标注。
        coarse_candidates = []
        for yaw_rad in np.linspace(-np.pi, np.pi, 72, endpoint=False):
            support, residual, view_index = self._critical_region_evidence(
                clouds, critical_vertices, body_center, float(yaw_rad), view_weights=view_weights
            )
            coarse_candidates.append((support, -residual, float(yaw_rad), body_center, view_index))
        coarse_candidates.sort(reverse=True)

        translated_candidates = []
        translation_offsets = np.array(
            [[dx, dy, dz] for dx in (-0.008, -0.004, 0.0, 0.004, 0.008)
             for dy in (-0.008, -0.004, 0.0, 0.004, 0.008)
             for dz in (-0.003, 0.0, 0.003)],
            dtype=np.float64,
        )
        for _, _, yaw_rad, _, _ in coarse_candidates[:4]:
            for offset in translation_offsets:
                center = body_center + offset
                support, residual, view_index = self._critical_region_evidence(
                    clouds, critical_vertices, center, yaw_rad, view_weights=view_weights
                )
                translated_candidates.append((support, -residual, yaw_rad, center, view_index))
        translated_candidates.sort(reverse=True)

        refined_candidates = []
        for _, _, candidate_yaw, candidate_center, _ in translated_candidates[:4]:
            for yaw_rad in candidate_yaw + np.arange(-0.12, 0.1201, 0.005):
                yaw_rad = float(np.arctan2(np.sin(yaw_rad), np.cos(yaw_rad)))
                support, residual, view_index = self._critical_region_evidence(
                    clouds, critical_vertices, candidate_center, yaw_rad, view_weights=view_weights
                )
                refined_candidates.append((support, -residual, yaw_rad, candidate_center, view_index))
        refined_candidates.sort(reverse=True)
        _, _, yaw_rad, _, _ = refined_candidates[0]
        rotation = self._rotation_z(yaw_rad)
        # 关键区平移搜索的目的只是让细小把手不因杯身中心的初估偏差而漏检，
        # 不能直接拿来输出对象原点：单个可见把手表面会把它拉向相机。最终平移
        # 改用全 CAD 表面的稳健点到面细化，且从杯身拟合结果启动，保留多视角
        # 杯身对位置的约束。
        mesh_matcher = GenericMeshCadMatcher()
        mesh_center_object = (model.vertices_object_m.min(axis=0) + model.vertices_object_m.max(axis=0)) / 2.0
        local_surface = mesh_matcher._surface_samples(model) - mesh_center_object
        _, mesh_center_base, _ = mesh_matcher._fit_translation(
            all_points,
            local_surface,
            yaw_rad,
            body_center + rotation @ mesh_center_object,
        )
        center = mesh_center_base - rotation @ mesh_center_object
        # 关键区的第一次搜索允许把手局部平移，是为了避免漏检；现在已有由完整
        # CAD 表面给出的对象中心，应在这个中心重新比较所有 yaw。否则“局部把手
        # 被拉向相机时的最高分”可能残留为最终朝向，尤其是在两个候选只差几度时。
        recentered_coarse = []
        for candidate_yaw in np.linspace(-np.pi, np.pi, 72, endpoint=False):
            candidate_support, candidate_residual, candidate_view = self._critical_region_evidence(
                clouds, critical_vertices, center, float(candidate_yaw), view_weights=view_weights
            )
            recentered_coarse.append((candidate_support, -candidate_residual, float(candidate_yaw), candidate_view))
        recentered_coarse.sort(reverse=True)
        _, _, recentered_yaw, _ = recentered_coarse[0]
        recentered_fine = []
        for candidate_yaw in recentered_yaw + np.arange(-0.12, 0.1201, 0.005):
            candidate_yaw = float(np.arctan2(np.sin(candidate_yaw), np.cos(candidate_yaw)))
            candidate_support, candidate_residual, candidate_view = self._critical_region_evidence(
                clouds, critical_vertices, center, candidate_yaw, view_weights=view_weights
            )
            recentered_fine.append((candidate_support, -candidate_residual, candidate_yaw, candidate_view))
        recentered_fine.sort(reverse=True)
        _, _, yaw_rad, _ = recentered_fine[0]
        rotation = self._rotation_z(yaw_rad)
        _, mesh_center_base, _ = mesh_matcher._fit_translation(
            all_points,
            local_surface,
            yaw_rad,
            body_center + rotation @ mesh_center_object,
        )
        center = mesh_center_base - rotation @ mesh_center_object
        support, critical_residual, evidence_view = self._critical_region_evidence(
            clouds, critical_vertices, center, yaw_rad, view_weights=view_weights
        )
        alternative_support = max(
            (
                self._critical_region_evidence(
                    clouds, critical_vertices, center, float(candidate_yaw), view_weights=view_weights
                )[0]
                for candidate_yaw in np.linspace(-np.pi, np.pi, 72, endpoint=False)
                if self._yaw_distance(float(candidate_yaw), yaw_rad) >= 0.35
            ),
            default=0.0,
        )
        support_margin = float(support - alternative_support)
        # 只有两条以上腕部基线都支持的 CAD 关键夹持区才能驱动最终姿态。固定
        # 相机在此阶段完全不参与评分；不足就继续主动观察或安全拒绝，而非回退
        # 到远距离粗姿态。物理双侧相对面接触仍是后续不可省略的最后门禁。
        support_requirement = 0.08
        # 两帧腕部时，6% 差值用于避免圆柱杯身的局部贴合伪装成把手；取得第三
        # 条独立腕部基线后，多个方位都支持同一候选本身已是额外约束，可将差值
        # 门槛收紧到仍可区分候选的 4%。这里完全不使用固定相机作为放行依据。
        margin_requirement = 0.04 if len(observations) >= 3 else 0.06
        if support < support_requirement or support_margin < margin_requirement:
            raise RuntimeError(
                "马克杯把手抓取关键区证据不足："
                f"覆盖率={support:.3f}，替代朝向差值={support_margin:.3f}，"
                f"最佳观察帧={evidence_view}"
            )

        distances = CylinderCadMatcher._surface_distance(all_points, center, self.body_radius_m, model.height_m)
        body_residual = self._robust_cost(distances)
        inlier_ratio = float(np.mean(distances < 0.008))
        body_confidence = float(np.clip((1.0 - body_residual / 0.012) * min(1.0, inlier_ratio / 0.45), 0.0, 1.0))
        # support 已按主动观察可见性折减；0.24 是该有效覆盖率达到“充分”的
        # 标尺。最终置信度只来自腕部的杯身与关键夹持区几何证据。
        critical_confidence = float(np.clip(support / 0.24, 0.0, 1.0))
        geometry_confidence = float(min(body_confidence, critical_confidence))
        if body_residual > 0.012 or inlier_ratio < 0.35 or geometry_confidence < 0.38:
            raise RuntimeError(
                f"马克杯多视角 CAD 配准置信度不足：杯身残差={body_residual:.4f}m，"
                f"内点率={inlier_ratio:.3f}，关键区覆盖={support:.3f}，几何置信度={geometry_confidence:.3f}"
            )
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = center
        return CadPoseEstimate(
            model_id=model.model_id,
            position_base_m=center,
            orientation_xyzw=np.array([0.0, 0.0, np.sin(yaw_rad / 2.0), np.cos(yaw_rad / 2.0)]),
            transform_base_object=transform,
            surface_residual_m=body_residual,
            inlier_ratio=inlier_ratio,
            segmentation_confidence=float(np.mean(confidences)),
            geometry_confidence=geometry_confidence,
            axial_yaw_observable=True,
            cylinder_axis_base=np.array([0.0, 0.0, 1.0]),
            point_count=len(all_points),
            matching_method="mug_handle_wrist_multiview_grasp_region_surface_refinement",
            object_yaw_rad=yaw_rad,
            critical_region_support=support,
            critical_region_margin=support_margin,
        )


class GenericMeshCadMatcher:
    """基于 CAD 表面模板的通用 6D 配准器。

    该实现复用工业视觉常见的“局部几何特征全局配准 + 点到面 ICP”流程：CAD
    网格均匀采样成模板，FPFH/RANSAC 产生不依赖桌面朝向的 6D 初值，随后以
    多尺度 point-to-plane ICP 细化。最终只对相机实际可见的 CAD 表面评分，避免
    背面和重复杯身把带把手物体拉到错误朝向。
    """

    _VOXEL_SCALES_M = (0.008, 0.004, 0.0025)
    _MODEL_CACHE: dict[str, tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]] = {}

    @staticmethod
    def _quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
        quaternion = np.empty(4, dtype=np.float64)
        # Numerically stable matrix-to-quaternion conversion, output convention xyzw.
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (rotation[2, 1] - rotation[1, 2]) / scale
            qy = (rotation[0, 2] - rotation[2, 0]) / scale
            qz = (rotation[1, 0] - rotation[0, 1]) / scale
        else:
            index = int(np.argmax(np.diag(rotation)))
            if index == 0:
                scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                qw, qx = (rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale
                qy, qz = (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale
            elif index == 1:
                scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                qw, qx = (rotation[0, 2] - rotation[2, 0]) / scale, (rotation[0, 1] + rotation[1, 0]) / scale
                qy, qz = 0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale
            else:
                scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                qw, qx = (rotation[1, 0] - rotation[0, 1]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale
                qy, qz = (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale
        quaternion[:] = (qx, qy, qz, qw)
        return quaternion / np.linalg.norm(quaternion)

    # Kept as small compatibility helpers for the established mug matcher.  The
    # generic path below no longer uses this yaw-only refinement as its primary
    # estimator.
    @staticmethod
    def _surface_samples(model: CadModel) -> np.ndarray:
        triangles = model.vertices_object_m[model.triangles_object]
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
        if not np.all(np.isfinite(areas)) or float(areas.sum()) <= 1e-12:
            raise RuntimeError(f"通用 CAD 网格表面积非法：{model.mesh_path}")
        count = min(900, max(300, len(triangles) * 4))
        generator = np.random.default_rng(20260923)
        chosen = generator.choice(len(triangles), size=count, p=areas / areas.sum())
        first, second = generator.random(count), generator.random(count)
        root = np.sqrt(first)
        barycentric = np.column_stack((1.0 - root, root * (1.0 - second), root * second))
        return (triangles[chosen] * barycentric[:, :, None]).sum(axis=1)

    @staticmethod
    def _nearest_surface(points: np.ndarray, surface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        squared = ((points[:, None, :] - surface[None, :, :]) ** 2).sum(axis=2)
        indices = np.argmin(squared, axis=1)
        return np.sqrt(squared[np.arange(len(points)), indices]), surface[indices]

    def _fit_translation(
        self, points: np.ndarray, local_surface: np.ndarray, yaw_rad: float, initial_center: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        rotation = self._rotation_z(yaw_rad)
        center = initial_center.copy()
        for _ in range(5):
            surface = local_surface @ rotation.T + center
            _, nearest = self._nearest_surface(points, surface)
            update = np.clip(np.median(points - nearest, axis=0), -0.025, 0.025)
            center += update
            if float(np.linalg.norm(update)) < 2e-4:
                break
        distances, _ = self._nearest_surface(points, local_surface @ rotation.T + center)
        return self._trimmed_mean(distances), center, distances

    @staticmethod
    def _point_cloud(points: np.ndarray, voxel_m: float) -> o3d.geometry.PointCloud:
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
        cloud = cloud.voxel_down_sample(voxel_m)
        if len(cloud.points) < 40:
            raise RuntimeError("通用 CAD 配准的有效点云不足")
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_m * 4.0, max_nn=48))
        cloud.normalize_normals()
        return cloud

    @classmethod
    def _model_template(cls, model: CadModel) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        cache_key = str(model.mesh_path.resolve())
        cached = cls._MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(model.vertices_object_m),
            o3d.utility.Vector3iVector(model.triangles_object.astype(np.int32)),
        )
        mesh.compute_vertex_normals()
        sampled = mesh.sample_points_uniformly(number_of_points=5000, use_triangle_normal=True)
        template = sampled.voxel_down_sample(cls._VOXEL_SCALES_M[0])
        template.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.032, max_nn=80))
        template.normalize_normals()
        features = o3d.pipelines.registration.compute_fpfh_feature(
            template, o3d.geometry.KDTreeSearchParamHybrid(radius=0.040, max_nn=120)
        )
        cls._MODEL_CACHE[cache_key] = (template, features)
        return template, features

    @staticmethod
    def _fpfh(cloud: o3d.geometry.PointCloud, voxel_m: float) -> o3d.pipelines.registration.Feature:
        return o3d.pipelines.registration.compute_fpfh_feature(
            cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_m * 8.0, max_nn=120)
        )

    @staticmethod
    def _trimmed_mean(values: np.ndarray, fraction: float = 0.72) -> float:
        if len(values) == 0:
            return float("inf")
        count = max(12, int(len(values) * fraction))
        return float(np.partition(values, count - 1)[:count].mean())

    @staticmethod
    def _rotation_z(angle: float) -> np.ndarray:
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])

    @staticmethod
    def _critical_vertices(model: CadModel) -> np.ndarray | None:
        templates = [item for item in model.grasp_templates if item.opposing_contact_pair is not None]
        if not templates:
            return None
        template = max(templates, key=lambda item: item.quality)
        pair = template.opposing_contact_pair
        midpoint = (pair.positive_surface.center_object_m + pair.negative_surface.center_object_m) / 2.0
        radius = float(np.clip(model.height_m * 0.36, 0.020, 0.045))
        vertices = model.vertices_object_m
        selected = vertices[np.linalg.norm(vertices - midpoint, axis=1) <= radius]
        return selected[:: max(1, len(selected) // 220)] if len(selected) >= 24 else None

    @staticmethod
    def _distances(query: np.ndarray, target_cloud: o3d.geometry.PointCloud) -> np.ndarray:
        query_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query))
        return np.asarray(query_cloud.compute_point_cloud_distance(target_cloud), dtype=np.float64)

    def _coarse_transforms(
        self, template: o3d.geometry.PointCloud, template_features: o3d.pipelines.registration.Feature,
        observed: o3d.geometry.PointCloud,
    ) -> list[tuple[np.ndarray, int]]:
        """Return deterministic, independent global FPFH pose hypotheses."""
        observed_features = self._fpfh(observed, self._VOXEL_SCALES_M[0])
        hypotheses: list[tuple[np.ndarray, int]] = []
        for seed_offset in range(4):
            # Open3D RANSAC otherwise consumes a process-global random stream,
            # making a saved RGB-D episode produce a different pose on replay.
            o3d.utility.random.seed(20261010 + seed_offset)
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                template, observed, template_features, observed_features, True, 0.020,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.85),
                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.020)],
                o3d.pipelines.registration.RANSACConvergenceCriteria(40000, 0.999),
            )
            # A fixed overview camera may see only a narrow RGB-D silhouette;
            # four non-coplanar FPFH correspondences are sufficient for a 6D
            # seed, while ICP/visible-surface validation below decides whether
            # it is usable.  Wrist fusion still needs stronger evidence.
            if len(result.correspondence_set) >= 4:
                hypotheses.append((np.asarray(result.transformation), len(result.correspondence_set)))
        if not hypotheses:
            raise RuntimeError("通用 CAD FPFH/RANSAC 未得到足够的全局几何对应")
        return hypotheses

    def _refine(
        self, model: CadModel, points: np.ndarray, initial: np.ndarray,
    ) -> tuple[np.ndarray, float, float, int]:
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(model.vertices_object_m),
            o3d.utility.Vector3iVector(model.triangles_object.astype(np.int32)),
        )
        mesh.compute_vertex_normals()
        source_full = mesh.sample_points_uniformly(number_of_points=6500, use_triangle_normal=True)
        transform = initial.copy()
        fitness, rmse = 0.0, float("inf")
        for voxel in self._VOXEL_SCALES_M:
            source = source_full.voxel_down_sample(voxel)
            target = self._point_cloud(points, voxel)
            result = o3d.pipelines.registration.registration_icp(
                source, target, voxel * 3.2, transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
            )
            transform, fitness, rmse = np.asarray(result.transformation), float(result.fitness), float(result.inlier_rmse)
        return transform, fitness, rmse, len(source_full.points)

    def _critical_evidence(
        self, model: CadModel, transform: np.ndarray, clouds: list[np.ndarray], target_cloud: o3d.geometry.PointCloud,
    ) -> tuple[float | None, float | None, int | None]:
        critical = self._critical_vertices(model)
        if critical is None:
            return None, None, None
        transformed = critical @ transform[:3, :3].T + transform[:3, 3]
        per_view_support = [
            float(np.mean(self._distances(transformed, self._point_cloud(cloud, 0.0025)) <= 0.012))
            for cloud in clouds
        ]
        support = max(per_view_support)
        opposite = transform.copy()
        opposite[:3, :3] = transform[:3, :3] @ self._rotation_z(np.pi)
        alternate = critical @ opposite[:3, :3].T + opposite[:3, 3]
        alternative_support = float(np.mean(self._distances(alternate, target_cloud) <= 0.012))
        return support, float(support - alternative_support), sum(value >= 0.08 for value in per_view_support)

    def _match_points(
        self,
        model: CadModel,
        points: np.ndarray,
        clouds: list[np.ndarray],
        segmentation_confidence: float,
        *,
        enforce_grasp_evidence: bool = True,
    ) -> CadPoseEstimate:
        if model.geometry_type != "mesh":
            raise ValueError("GenericMeshCadMatcher 仅处理 geometry_type=mesh")
        observed = self._point_cloud(points, self._VOXEL_SCALES_M[0])
        template, template_features = self._model_template(model)
        coarse_hypotheses = self._coarse_transforms(template, template_features, observed)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(model.vertices_object_m),
            o3d.utility.Vector3iVector(model.triangles_object.astype(np.int32)),
        )
        observed_fine = self._point_cloud(points, 0.0025)
        observed_points = np.asarray(observed_fine.points)
        # The cylindrical part of a mug has repeated geometry.  Branching four
        # CAD-local yaw hypotheses lets the handle select the correct global
        # solution after RANSAC without assuming that the object is upright.
        # This is a bounded, deterministic form of industrial pose-hypothesis
        # verification rather than the old exhaustive x/y/yaw grid search.
        candidates = []
        for initial, correspondence_count in coarse_hypotheses:
            for yaw_offset in (0.0, np.pi / 2.0, np.pi, -np.pi / 2.0):
                local_yaw = np.eye(4)
                local_yaw[:3, :3] = self._rotation_z(yaw_offset)
                transform, icp_fitness, icp_rmse, sample_count = self._refine(model, points, initial @ local_yaw)
                rendered = mesh.sample_points_uniformly(number_of_points=4000)
                rendered.transform(transform)
                rendered_points = np.asarray(rendered.points)
                observed_to_model = self._distances(observed_points, rendered)
                model_to_observed = self._distances(rendered_points, observed_fine)
                residual = self._trimmed_mean(observed_to_model)
                visible_support = float(np.mean(model_to_observed <= 0.010))
                inlier_ratio = float(np.mean(observed_to_model <= 0.008))
                critical_support, critical_margin, critical_baseline_count = self._critical_evidence(
                    model, transform, clouds, observed_fine
                )
                critical_ok = critical_support is None or (
                    critical_support >= 0.08
                    and critical_margin is not None
                    and critical_margin >= 0.06
                    and critical_baseline_count >= 2
                )
                score = 0.48 * inlier_ratio + 0.32 * visible_support + 0.20 * icp_fitness - residual / 0.020
                if not critical_ok:
                    score -= 0.75
                candidates.append((score, transform, icp_fitness, icp_rmse, sample_count, residual, visible_support,
                                   inlier_ratio, critical_support, critical_margin, critical_baseline_count,
                                   correspondence_count))
        (_, transform, icp_fitness, icp_rmse, sample_count, residual, visible_support,
         inlier_ratio, critical_support, critical_margin, critical_baseline_count,
         correspondence_count) = max(candidates, key=lambda item: item[0])
        if enforce_grasp_evidence and critical_support is not None and (
            critical_support < 0.08
            or critical_margin is None
            or critical_margin < 0.06
            or critical_baseline_count < 2
        ):
            raise RuntimeError(
                "通用 CAD 抓取关键区证据不足："
                f"覆盖率={critical_support:.3f}，替代朝向差值={critical_margin:.3f}，"
                f"支持腕部基线数={critical_baseline_count}"
            )
        # ICP fitness is measured CAD→partial-observation, hence it is
        # intentionally low for an occluded but geometrically excellent view.
        # Use the bidirectional visible-surface score as the acceptance signal;
        # retain raw ICP fitness in diagnostics for audit rather than making
        # partial visibility look like a failed registration.
        confidence = float(np.clip(
            (0.60 * inlier_ratio + 0.40 * visible_support) * (1.0 - residual / 0.018), 0.0, 1.0
        ))
        if residual > 0.012 or inlier_ratio < 0.42 or visible_support < 0.20 or confidence < 0.30:
            raise RuntimeError(
                f"通用 CAD 表面配准置信度不足：残差={residual:.4f}m，内点率={inlier_ratio:.3f}，可见支持={visible_support:.3f}"
            )
        yaw = float(np.arctan2(transform[1, 0], transform[0, 0]))
        return CadPoseEstimate(
            model_id=model.model_id, position_base_m=transform[:3, 3],
            orientation_xyzw=self._quaternion_xyzw(transform[:3, :3]), transform_base_object=transform,
            surface_residual_m=residual, inlier_ratio=inlier_ratio, segmentation_confidence=segmentation_confidence,
            geometry_confidence=confidence, axial_yaw_observable=True, cylinder_axis_base=transform[:3, 2],
            point_count=len(points), matching_method="generic_mesh_fpfh_ransac_multiscale_icp_visible_surface_refinement",
            object_yaw_rad=yaw, critical_region_support=critical_support, critical_region_margin=critical_margin,
            registration_diagnostics={"coarse_method": "fpfh_ransac", "coarse_correspondence_count": correspondence_count,
                                      "coarse_hypothesis_count": len(candidates),
                                      "icp_method": "multiscale_point_to_plane", "icp_fitness": icp_fitness,
                                      "icp_rmse_m": icp_rmse, "visible_surface_support": visible_support,
                                      "critical_region_baseline_count": critical_baseline_count,
                                      "cad_surface_sample_count": sample_count},
        )

    def _coarse_pose_from_points(
        self,
        model: CadModel,
        points: np.ndarray,
        frame: UnifiedCameraFrame,
        mask: np.ndarray,
        segmentation_confidence: float,
        reason: str,
    ) -> CadPoseEstimate:
        """Generate a conservative geometric seed when one overview frame is sparse.

        This is deliberately not a graspable result: it only allows active
        wrist observation to start.  The final pose is always rebuilt by the
        FPFH/RANSAC/ICP multi-view path and its two-baseline grasp gate.
        """
        center = (np.min(points, axis=0) + np.max(points, axis=0)) / 2.0
        triangles = model.vertices_object_m[model.triangles_object]
        triangle_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        triangle_normals /= np.linalg.norm(triangle_normals, axis=1, keepdims=True)
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
        generator = np.random.default_rng(20260923)
        chosen = generator.choice(len(triangles), size=1600, p=areas / areas.sum())
        first, second = generator.random(1600), generator.random(1600)
        root = np.sqrt(first)
        barycentric = np.column_stack((1.0 - root, root * (1.0 - second), root * second))
        surface = (triangles[chosen] * barycentric[:, :, None]).sum(axis=1)
        normals = triangle_normals[chosen]
        # This bounded 2.5D score is a *seed only* for moving the wrist toward
        # an otherwise feature-poor overview.  It does not emit a graspable
        # pose or participate in final selection, which remains full 6D
        # FPFH/RANSAC + multi-scale ICP across wrist RGB-D frames.
        candidates = []
        for yaw_candidate in np.linspace(-np.pi, np.pi, num=48, endpoint=False):
            rotation = self._rotation_z(float(yaw_candidate))
            world_surface = surface @ rotation.T + center
            world_normals = normals @ rotation.T
            # Approximate a CAD depth render by keeping only front-facing mesh
            # samples.  Unlike the old all-surface Chamfer seed, this lets the
            # overview image distinguish which side contains the handle.
            camera_position_base_m = frame.base_from_camera[:3, 3]
            facing = np.einsum("ij,ij->i", world_normals, camera_position_base_m - world_surface) > 0.02
            visible_surface = surface[facing]
            if len(visible_surface) < 80:
                visible_surface = surface
            residual, candidate_center, _ = self._fit_translation(
                points, visible_surface, float(yaw_candidate), center
            )
            rendered = visible_surface @ rotation.T + candidate_center
            homogeneous = np.column_stack((rendered, np.ones(len(rendered))))
            camera = homogeneous @ np.linalg.inv(frame.base_from_camera).T
            depth = -camera[:, 2]
            rows = np.rint(frame.intrinsic[1, 2] - camera[:, 1] * frame.intrinsic[1, 1] / depth).astype(int)
            columns = np.rint(frame.intrinsic[0, 2] + camera[:, 0] * frame.intrinsic[0, 0] / depth).astype(int)
            inside = (
                np.isfinite(depth) & (depth > 0.10)
                & (rows >= 0) & (rows < mask.shape[0]) & (columns >= 0) & (columns < mask.shape[1])
            )
            rows, columns, depth = rows[inside], columns[inside], depth[inside]
            observed = frame.depth[rows, columns] if len(rows) else np.empty(0)
            aligned = np.isfinite(observed) & (observed > 0.10) & mask[rows, columns]
            image_residual = self._trimmed_mean(np.abs(observed[aligned] - depth[aligned])) if np.any(aligned) else 0.10
            coverage = float(np.count_nonzero(aligned) / max(1, len(visible_surface)))
            candidates.append((residual + image_residual - 0.010 * coverage, residual, float(yaw_candidate), candidate_center))
        _, residual, yaw, center = min(candidates, key=lambda item: item[0])
        rotation = self._rotation_z(yaw)
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, center
        return CadPoseEstimate(
            model_id=model.model_id,
            position_base_m=center,
            orientation_xyzw=self._quaternion_xyzw(rotation),
            transform_base_object=transform,
            surface_residual_m=residual,
            inlier_ratio=0.0,
            segmentation_confidence=segmentation_confidence,
            geometry_confidence=0.05,
            axial_yaw_observable=False,
            cylinder_axis_base=rotation[:, 2],
            point_count=len(points),
            matching_method="generic_mesh_overview_geometric_seed_for_active_observation",
            object_yaw_rad=yaw,
            registration_diagnostics={
                "coarse_seed_only": True,
                "fallback_reason": reason,
                "overview_surface_seed_yaw_hypothesis_count": len(candidates),
            },
        )

    def match(self, model: CadModel, segmentation: SegmentationResult, frame: UnifiedCameraFrame) -> CadPoseEstimate:
        points = _points_from_mask(frame, segmentation.mask)
        # A fixed-camera mesh pose is only a coarse ROI/approach seed.  It must
        # not be allowed to satisfy the two-wrist-baseline grasp gate by itself;
        # ``match_multiview`` below is the sole producer of a graspable pose.
        try:
            estimate = self._match_points(
                model, points, [points], segmentation.confidence, enforce_grasp_evidence=False
            )
            # 当前控制场景是已知桌面工位。固定 RGB-D 的局部杯身视图有时会令
            # 全局 FPFH 把 CAD 倾倒到错误 6D 局部极值；这类结果不能拿来决定
            # 中性观察位的 TCP 方向。仅在“竖直且中心接近观测包围盒”时采用它；
            # 否则保留相同点云生成的保守桌面几何种子，最终 6D/yaw 仍由腕部
            # ROI 表面证据决定。
            observed_center = (np.min(points, axis=0) + np.max(points, axis=0)) / 2.0
            vertical_alignment = float(np.dot(estimate.transform_base_object[:3, 2], np.array([0.0, 0.0, 1.0])))
            center_error = float(np.linalg.norm(estimate.position_base_m - observed_center))
            if vertical_alignment < 0.99 or center_error > 0.030:
                raise RuntimeError(
                    "固定相机 FPFH 全局 pose 不满足桌面粗定位一致性："
                    f"竖直对齐={vertical_alignment:.3f}，中心偏差={center_error:.4f}m"
                )
            return estimate
        except RuntimeError as error:
            return self._coarse_pose_from_points(
                model, points, frame, segmentation.mask, segmentation.confidence, str(error)
            )

    def match_multiview(self, model: CadModel, observations: list[tuple[SegmentationResult, UnifiedCameraFrame]]) -> CadPoseEstimate:
        if len(observations) < 2:
            raise ValueError("多视角 mesh 配准至少需要两帧 RGB-D 观测")
        clouds = [_points_from_mask(frame, segmentation.mask) for segmentation, frame in observations]
        balanced = [cloud[:: max(1, len(cloud) // 900)] for cloud in clouds]
        # 通用全局配准只负责把对象送入近距抓取工作区。两片夹持面的
        # 安全门禁由 GraspRoiMatcher 在下游单独执行，不能把它混作整体
        # CAD pose 的失败原因。
        return self._match_points(
            model,
            np.concatenate(balanced, axis=0),
            clouds,
            float(np.mean([item[0].confidence for item in observations])),
            enforce_grasp_evidence=False,
        )


class GraspRoiMatcher:
    """从 ``opposing_contact_pair`` 派生的通用抓取局部表面 matcher。

    它不识别物体类别，也不读取仿真真值：全局 CAD pose 给出物体附近的工作区，
    两片由 grasp JSON 标注的接触面则在腕部 RGB-D 中独立匹配。为处理杯身等
    旋转对称主体，先保留一组 CAD-local yaw 假设，再对每一假设做受限的局部
    刚体精修。最终只有正、负接触面均有观测支持时才输出抓取 pose。
    """

    _SUPPORT_DISTANCE_M = 0.009
    _CORRESPONDENCE_DISTANCE_M = 0.018
    _MIN_SURFACE_SUPPORT = 0.16
    # RGB-D 离散化、腕部外参和薄把手的 CAD 三角面会在单侧留下约厘米级
    # 残差；超过 12 mm 则不再把它视为可安全夹持的同一接触面。
    _MAX_SURFACE_RESIDUAL_M = 0.012
    # 全局表面配准只承诺把机械臂带入近距工作区；局部抓取面允许在该工作区内
    # 修正 2.5 cm，最终仍由双接触区残差而非粗中心决定是否闭爪。
    _MAX_LOCAL_TRANSLATION_M = 0.025
    _MAX_LOCAL_ROTATION_DEG = 10.0
    _YAW_HYPOTHESES = 24

    @staticmethod
    def _rotation_angle_deg(rotation: np.ndarray) -> float:
        cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
        return float(np.rad2deg(np.arccos(cosine)))

    @staticmethod
    def _nearest(query: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        squared = ((query[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
        indices = np.argmin(squared, axis=1)
        return np.sqrt(squared[np.arange(len(query)), indices]), target[indices], indices

    @staticmethod
    def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return the rigid correction that maps already-world ``source`` to target."""
        source_center, target_center = source.mean(axis=0), target.mean(axis=0)
        covariance = (source - source_center).T @ (target - target_center)
        left, _, right = np.linalg.svd(covariance)
        rotation = right.T @ left.T
        if np.linalg.det(rotation) < 0.0:
            right[-1, :] *= -1.0
            rotation = right.T @ left.T
        correction = np.eye(4)
        correction[:3, :3] = rotation
        correction[:3, 3] = target_center - rotation @ source_center
        return correction

    @staticmethod
    def _roi_vertices(model: CadModel, surface: ContactSurface) -> np.ndarray:
        """Extract a CAD patch around one annotated contact surface.

        The annotation gives a point and normal, not mesh face indices.  A
        thin normal slab plus a height-scaled tangent envelope works for any
        CAD carrying the existing generic contact-pair schema.
        """
        vertices = model.vertices_object_m
        if surface.patch_radius_m is not None:
            triangles = vertices[model.triangles_object]
            areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
            generator = np.random.default_rng(20261013)
            indices = generator.choice(len(triangles), 12000, p=areas / areas.sum())
            u, v = generator.random((2, len(indices)))
            root = np.sqrt(u)
            weights = np.column_stack((1.0 - root, root * (1.0 - v), root * v))
            vertices = np.concatenate((vertices, (triangles[indices] * weights[:, :, None]).sum(axis=1)))
        delta = vertices - surface.center_object_m
        normal_distance = np.abs(delta @ surface.normal_object)
        tangent_distance = np.linalg.norm(
            delta - np.outer(delta @ surface.normal_object, surface.normal_object), axis=1
        )
        slab_m = float(np.clip(model.height_m * 0.055, 0.0035, 0.0065))
        radius_m = surface.patch_radius_m or float(np.clip(model.height_m * 0.38, 0.024, 0.045))
        selected = vertices[(normal_distance <= slab_m) & (tangent_distance <= radius_m)]
        if len(selected) < 16:
            raise RuntimeError(f"抓取 ROI CAD 面采样不足：{surface.region_id} 仅有 {len(selected)} 个顶点")
        return selected[:: max(1, len(selected) // 120)]

    @staticmethod
    def _trimmed_mean(values: np.ndarray, fraction: float = 0.50) -> float:
        if len(values) == 0:
            return float("inf")
        count = max(8, int(len(values) * fraction))
        return float(np.partition(values, count - 1)[:count].mean())

    def _surface_evidence(
        self, transformed_surface: np.ndarray, clouds: list[np.ndarray]
    ) -> tuple[float, float, list[float]]:
        supports: list[float] = []
        residuals: list[float] = []
        for cloud in clouds:
            distances, _, _ = self._nearest(transformed_surface, cloud)
            supports.append(float(np.mean(distances <= self._SUPPORT_DISTANCE_M)))
            residuals.append(self._trimmed_mean(distances))
        return max(supports, default=0.0), min(residuals, default=float("inf")), supports

    def _refine_candidate(
        self, initial: np.ndarray, object_roi: np.ndarray, observed: np.ndarray
    ) -> np.ndarray | None:
        transform = initial.copy()
        for _ in range(6):
            world_roi = object_roi @ transform[:3, :3].T + transform[:3, 3]
            distances, nearest, _ = self._nearest(world_roi, observed)
            keep = distances <= self._CORRESPONDENCE_DISTANCE_M
            if int(np.count_nonzero(keep)) < 18:
                return None
            correction = self._kabsch(world_roi[keep], nearest[keep])
            transform = correction @ transform
            # ``transform @ inv(initial)`` 的平移分量会包含“绕 CAD 原点
            # 旋转”带来的基座原点项；这里的局部收敛约束应比较物体中心位置，
            # 而非该复合矩阵的平移列。
            relative_rotation = transform[:3, :3] @ initial[:3, :3].T
            if (
                float(np.linalg.norm(transform[:3, 3] - initial[:3, 3])) > self._MAX_LOCAL_TRANSLATION_M
                or self._rotation_angle_deg(relative_rotation) > self._MAX_LOCAL_ROTATION_DEG
            ):
                return None
            if (
                float(np.linalg.norm(correction[:3, 3])) < 2e-4
                and self._rotation_angle_deg(correction[:3, :3]) < 0.15
            ):
                break
        return transform

    def refine(
        self,
        model: CadModel,
        template: GraspTemplate,
        global_pose: CadPoseEstimate,
        observations: list[tuple[SegmentationResult, UnifiedCameraFrame]],
    ) -> GraspRoiEstimate:
        clouds = [
            _points_from_mask(frame, segmentation.mask)[:: max(1, segmentation.pixel_count // 850)]
            for segmentation, frame in observations
        ]
        return self.refine_clouds(model, template, global_pose, clouds)

    def refine_clouds(
        self,
        model: CadModel,
        template: GraspTemplate,
        global_pose: CadPoseEstimate,
        clouds: list[np.ndarray],
    ) -> GraspRoiEstimate:
        """Cloud-level entry point used by deterministic regression tests."""
        if len(clouds) < 2:
            raise RuntimeError("抓取 ROI 精配准至少需要两条腕部 RGB-D 基线")
        clouds = [np.asarray(cloud, dtype=np.float64) for cloud in clouds]
        if any(len(cloud) < 40 for cloud in clouds):
            raise RuntimeError("抓取 ROI 精配准的单帧有效点不足")
        positive_object = self._roi_vertices(model, template.opposing_contact_pair.positive_surface)
        negative_object = self._roi_vertices(model, template.opposing_contact_pair.negative_surface)
        object_roi = np.concatenate((positive_object, negative_object), axis=0)
        observed = np.concatenate(clouds, axis=0)
        if len(observed) < 80:
            raise RuntimeError("抓取 ROI 精配准的腕部有效点不足")

        candidates: list[tuple[float, np.ndarray, float, float, float, float, int, float]] = []
        # 局部平面本身可能近似对称。以整份 CAD 对腕部观测的解释程度作为
        # 抓取区候选的上下文约束，避免方形耳部等局部区域产生 90° 错配。
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(model.vertices_object_m),
            o3d.utility.Vector3iVector(model.triangles_object.astype(np.int32)),
        )
        context_scene = o3d.t.geometry.RaycastingScene()
        context_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        context_observed = observed[::max(1, len(observed) // 1800)]
        for yaw_offset in np.linspace(-np.pi, np.pi, self._YAW_HYPOTHESES, endpoint=False):
            local_yaw = np.eye(4)
            local_yaw[:3, :3] = GenericMeshCadMatcher._rotation_z(float(yaw_offset))
            initial = global_pose.transform_base_object @ local_yaw
            refined = self._refine_candidate(initial, object_roi, observed)
            # 抓取区在单一腕部图像里经常只露出把手的一条外弧。此时最近邻
            # Kabsch 可能被部分曲面拉向错误的小旋转；原始 CAD yaw 假设本身
            # 仍须作为可解释候选进入双表面证据门禁，而不是被优化器静默丢弃。
            if refined is None:
                refined = initial
            positive = positive_object @ refined[:3, :3].T + refined[:3, 3]
            negative = negative_object @ refined[:3, :3].T + refined[:3, 3]
            positive_support, positive_residual, positive_per_view = self._surface_evidence(positive, clouds)
            negative_support, negative_residual, negative_per_view = self._surface_evidence(negative, clouds)
            supporting_views = sum(
                max(pos, neg) >= self._MIN_SURFACE_SUPPORT
                for pos, neg in zip(positive_per_view, negative_per_view)
            )
            pair_support = min(positive_support, negative_support)
            score = 1.20 * pair_support + 0.20 * (positive_support + negative_support) - (
                positive_residual + negative_residual
            ) / 0.030
            object_points = (context_observed - refined[:3, 3]) @ refined[:3, :3]
            context_distances = context_scene.compute_distance(o3d.core.Tensor(object_points.astype(np.float32))).numpy()
            context_support = float(np.mean(context_distances <= 0.006))
            score += 1.2 * context_support
            candidates.append((
                score, refined, positive_support, negative_support, positive_residual,
                negative_residual, supporting_views, float(yaw_offset),
            ))
        if not candidates:
            raise RuntimeError("抓取 ROI 精配准没有获得足够的 CAD--深度对应")
        candidates.sort(key=lambda item: item[0], reverse=True)
        (score, transform, positive_support, negative_support, positive_residual,
         negative_residual, supporting_views, selected_yaw_offset) = candidates[0]
        alternative_score = max(
            (item[0] for item in candidates[1:]
             if abs(float(np.arctan2(np.sin(item[7] - selected_yaw_offset), np.cos(item[7] - selected_yaw_offset)))) >= 0.35),
            default=score - 1.0,
        )
        margin = float(score - alternative_score)
        translation_correction = float(np.linalg.norm(
            transform[:3, 3] - global_pose.transform_base_object[:3, 3]
        ))
        rotation_correction = self._rotation_angle_deg(
            transform[:3, :3] @ global_pose.transform_base_object[:3, :3].T
        )
        if (
            positive_support < self._MIN_SURFACE_SUPPORT
            or negative_support < self._MIN_SURFACE_SUPPORT
            or supporting_views < 2
            or margin < 0.06
            or positive_residual > self._MAX_SURFACE_RESIDUAL_M
            or negative_residual > self._MAX_SURFACE_RESIDUAL_M
        ):
            raise RuntimeError(
                "抓取 ROI 证据不足："
                f"正面覆盖={positive_support:.3f}，负面覆盖={negative_support:.3f}，"
                f"支持视角={supporting_views}，候选差值={margin:.3f}，"
                f"残差=({positive_residual:.4f},{negative_residual:.4f})m"
            )
        confidence = float(np.clip(
            min(global_pose.geometry_confidence, (positive_support + negative_support) / 0.70)
            * (1.0 - max(positive_residual, negative_residual) / 0.012), 0.0, 1.0
        ))
        diagnostics = dict(global_pose.registration_diagnostics or {})
        diagnostics["grasp_roi"] = {
            "template_id": template.template_id,
            "positive_surface_region": template.opposing_contact_pair.positive_surface.region_id,
            "negative_surface_region": template.opposing_contact_pair.negative_surface.region_id,
            "positive_support": positive_support,
            "negative_support": negative_support,
            "positive_residual_m": positive_residual,
            "negative_residual_m": negative_residual,
            "supporting_view_count": supporting_views,
            "alternative_yaw_margin": margin,
            "translation_correction_m": translation_correction,
            "rotation_correction_deg": rotation_correction,
            "selected_yaw_offset_rad": selected_yaw_offset,
        }
        # 视觉匹配的离散 CAD yaw 与机器人 IK 的窄奇异区不应彼此绑死。只对
        # ±5° 的局部邻域重新计算两片接触面的独立深度支持；这样后续由 MoveIt
        # 选择可达姿态时仍然是“ROI 已验证”的 CAD pose，而不是无约束地旋转
        # 夹爪去碰运气。
        reachable_yaw_offsets: list[float] = []
        for yaw_delta in (0.0, -np.deg2rad(5.0), np.deg2rad(5.0), -np.deg2rad(10.0), np.deg2rad(10.0)):
            local_rotation = GenericMeshCadMatcher._rotation_z(float(yaw_delta))
            candidate_transform = transform.copy()
            candidate_transform[:3, :3] = transform[:3, :3] @ local_rotation
            candidate_positive = positive_object @ candidate_transform[:3, :3].T + candidate_transform[:3, 3]
            candidate_negative = negative_object @ candidate_transform[:3, :3].T + candidate_transform[:3, 3]
            candidate_positive_support, candidate_positive_residual, candidate_positive_views = self._surface_evidence(
                candidate_positive, clouds
            )
            candidate_negative_support, candidate_negative_residual, candidate_negative_views = self._surface_evidence(
                candidate_negative, clouds
            )
            candidate_view_count = sum(
                max(pos, neg) >= self._MIN_SURFACE_SUPPORT
                for pos, neg in zip(candidate_positive_views, candidate_negative_views)
            )
            if (
                candidate_positive_support >= self._MIN_SURFACE_SUPPORT
                and candidate_negative_support >= self._MIN_SURFACE_SUPPORT
                and candidate_view_count >= 2
                and candidate_positive_residual <= self._MAX_SURFACE_RESIDUAL_M
                and candidate_negative_residual <= self._MAX_SURFACE_RESIDUAL_M
            ):
                reachable_yaw_offsets.append(float(yaw_delta))
        diagnostics["grasp_roi"]["reachability_validated_yaw_offsets_rad"] = reachable_yaw_offsets
        pose = replace(
            global_pose,
            position_base_m=transform[:3, 3],
            orientation_xyzw=GenericMeshCadMatcher._quaternion_xyzw(transform[:3, :3]),
            transform_base_object=transform,
            geometry_confidence=confidence,
            axial_yaw_observable=True,
            cylinder_axis_base=transform[:3, 2],
            # 固定相机的 fallback seed 在此已被真正的双表面 ROI 精修取代；对外
            # 仍保持通用 CAD 管线的稳定名称，细节在 diagnostics 中可审计。
            matching_method=(
                "generic_mesh_fpfh_ransac_multiscale_icp_visible_surface_refinement"
                if global_pose.matching_method.startswith("generic_mesh_")
                else global_pose.matching_method
            ),
            object_yaw_rad=float(np.arctan2(transform[1, 0], transform[0, 0])),
            critical_region_support=min(positive_support, negative_support),
            critical_region_margin=margin,
            registration_diagnostics=diagnostics,
        )
        return GraspRoiEstimate(
            pose=pose,
            template_id=template.template_id,
            positive_support=positive_support,
            negative_support=negative_support,
            positive_residual_m=positive_residual,
            negative_residual_m=negative_residual,
            supporting_view_count=supporting_views,
            alternative_margin=margin,
            translation_correction_m=translation_correction,
            rotation_correction_deg=rotation_correction,
            selected_yaw_offset_rad=selected_yaw_offset,
            reachability_validated_yaw_offsets_rad=tuple(reachable_yaw_offsets),
        )


class CadMatcherDispatcher:
    """按已校验的 CAD ``geometry_type`` 选择匹配器。"""

    def __init__(self) -> None:
        self._matchers = {
            "cylinder": CylinderCadMatcher(),
            "box": BoxCadMatcher(),
            "mug_handle": MugHandleCadMatcher(),
            "mesh": GenericMeshCadMatcher(),
        }

    def match(self, model: CadModel, segmentation: SegmentationResult, frame: UnifiedCameraFrame) -> CadPoseEstimate:
        matcher = self._matchers.get(model.geometry_type)
        if matcher is None:
            raise ValueError(f"未实现 geometry_type={model.geometry_type} 的 CAD 匹配器")
        return matcher.match(model, segmentation, frame)

    def match_multiview(
        self,
        model: CadModel,
        observations: list[tuple[SegmentationResult, UnifiedCameraFrame]],
    ) -> CadPoseEstimate:
        matcher = self._matchers.get(model.geometry_type)
        if not isinstance(matcher, (GenericMeshCadMatcher, MugHandleCadMatcher)):
            raise ValueError(f"geometry_type={model.geometry_type} 尚未实现多视角 CAD 融合")
        if isinstance(matcher, MugHandleCadMatcher):
            return matcher.match_multiview(model, observations)
        return matcher.match_multiview(model, observations)


def grasp_pose_from_template(pose: CadPoseEstimate, template: GraspTemplate) -> tuple[np.ndarray, np.ndarray]:
    """将对象标注抓取框变换为 MoveIt ``panda_hand`` 目标。

    CAD 标注使用 MuJoCo 指尖的夹合坐标系。实际 ``hand`` 的闭合轴相对 MoveIt
    ``panda_hand`` 目标反向偏 45°，故目标须施加正向 45° 补偿。该符号已通过
    真实 MuJoCo 双指接触的轴对齐检查验证。
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
    # q_target = q_pad_grasp * q_z(+45°)，xyzw 格式。补偿后 MuJoCo 中实际
    # 手爪的 Y 轴（闭合轴）与 CAD 标注的相对夹持面法向平行。
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
