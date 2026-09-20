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
    destination_category: str
    confidence: float
    source: str

    def as_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "target_query": {"category": self.target_category, "attributes": {"color": self.target_color}},
            "destination_query": {"category": self.destination_category},
            "confidence": self.confidence,
            "source": self.source,
        }


class RuleVlmAdapter:
    """与未来真实 VLM 共用任务 JSON 的临时规则实现。"""

    supported_colors = {"红色": "red", "绿色": "green", "蓝色": "blue"}

    def infer(self, instruction: str) -> TaskIntent:
        selected = [chinese for chinese in self.supported_colors if chinese in instruction]
        if len(selected) != 1 or "杯" not in instruction or "托盘" not in instruction:
            raise ValueError("当前 CAD 基线仅支持“抓取一种颜色杯子并放到托盘”的文本指令")
        return TaskIntent(
            task="pick_and_place",
            target_category="cup",
            target_color=self.supported_colors[selected[0]],
            destination_category="tray",
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
            "blue": (blue > 105) & (blue > red * 1.25) & (blue > green * 1.15),
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
class GraspTemplate:
    template_id: str
    position_object_m: np.ndarray
    orientation_object_xyzw: np.ndarray
    approach_clearance_m: float
    lift_clearance_m: float
    required_gripper_width_m: float


@dataclass(frozen=True)
class CadModel:
    model_id: str
    category: str
    color: str
    mesh_path: Path
    vertices_object_m: np.ndarray
    radius_m: float
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
            "matching_method": "cad_cylinder_surface_sdf_grid_refinement",
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
        self._models = tuple(self._load_model(item) for item in raw["models"])

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

    def _load_model(self, raw: dict[str, object]) -> CadModel:
        mesh_path = self._root / str(raw["mesh"])
        vertices = self._load_obj_vertices(mesh_path)
        radial = np.linalg.norm(vertices[:, :2], axis=1)
        template_data = raw["grasp_templates"]
        templates = tuple(
            GraspTemplate(
                template_id=str(template["template_id"]),
                position_object_m=np.asarray(template["position_object_m"], dtype=np.float64),
                orientation_object_xyzw=np.asarray(template["orientation_object_xyzw"], dtype=np.float64),
                approach_clearance_m=float(template["approach_clearance_m"]),
                lift_clearance_m=float(template["lift_clearance_m"]),
                required_gripper_width_m=float(template["required_gripper_width_m"]),
            )
            for template in template_data  # type: ignore[union-attr]
        )
        attributes = raw["attributes"]  # type: ignore[assignment]
        symmetry = raw["symmetry"]  # type: ignore[assignment]
        return CadModel(
            model_id=str(raw["model_id"]),
            category=str(raw["category"]),
            color=str(attributes["color"]),  # type: ignore[index]
            mesh_path=mesh_path,
            vertices_object_m=vertices,
            radius_m=float(np.median(radial)),
            height_m=float(vertices[:, 2].max() - vertices[:, 2].min()),
            symmetry_type=str(symmetry["type"]),  # type: ignore[index]
            grasp_templates=templates,
        )

    def find(self, intent: TaskIntent) -> CadModel:
        candidates = [
            model for model in self._models if model.category == intent.target_category and model.color == intent.target_color
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"对象目录未找到唯一 CAD 模型：{intent.target_category}/{intent.target_color}")
        return candidates[0]

    def fixture_collision_positions(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(value["position_base_m"], dtype=np.float64)
            for name, value in self._fixture.items()
        }

    def tray_center_base_m(self) -> np.ndarray:
        return np.asarray(self._fixture["tray"]["position_base_m"], dtype=np.float64)

    def physical_placement_targets(self, model: CadModel, template: GraspTemplate) -> PhysicalPlacementTargets:
        """从静态托盘几何和 CAD/抓取模板推导释放位，不读取仿真真值。"""
        tray = self._fixture["tray"]
        center = np.asarray(tray["position_base_m"], dtype=np.float64)
        floor_z_offset = float(tray["floor_z_offset_m"])
        drop_clearance = float(tray["drop_clearance_m"])
        approach_clearance = float(tray["place_approach_clearance_m"])
        object_center = center + np.array([0.0, 0.0, floor_z_offset + model.height_m / 2.0 + drop_clearance])
        # 本阶段已限定对象竖直、抓取模板为顶部垂直接近；因此对象坐标的模板平移与基座轴对齐。
        gripper_release = object_center + template.position_object_m
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
        best_cost, best_center = float("inf"), initial.copy()
        offsets = np.arange(-span, span + step * 0.5, step)
        # 限制成本计算点数；被均匀抽样的点仍覆盖圆柱可见表面，保证 VNC 运行时延可控。
        sample = points[:: max(1, len(points) // 600)]
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    candidate = initial + np.array([dx, dy, dz])
                    value = self._robust_cost(self._surface_distance(sample, candidate, model.radius_m, model.height_m))
                    if value < best_cost:
                        best_cost, best_center = value, candidate
        return best_cost, best_center

    def match(self, model: CadModel, segmentation: SegmentationResult, frame: UnifiedCameraFrame) -> CadPoseEstimate:
        if model.symmetry_type != "axial":
            raise ValueError("当前首版仅支持轴对称圆柱 OBJ；通用网格匹配将在后续实现")
        points = _points_from_mask(frame, segmentation.mask)
        initial = np.median(points, axis=0)
        _, coarse = self._search(points, initial, step=0.005, span=0.050, model=model)
        _, center = self._search(points, coarse, step=0.001, span=0.006, model=model)
        distances = self._surface_distance(points, center, model.radius_m, model.height_m)
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
        )


def grasp_pose_from_template(pose: CadPoseEstimate, template: GraspTemplate) -> tuple[np.ndarray, np.ndarray]:
    """将对象坐标系下的 CAD 顶抓模板变换为基座系末端目标。"""
    rotation = pose.transform_base_object[:3, :3]
    position = pose.position_base_m + rotation @ template.position_object_m
    return position, template.orientation_object_xyzw.copy()
