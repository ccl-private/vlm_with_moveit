#!/usr/bin/env python3
"""核验同一通用控制管线的不同 CAD 回合，不将专用 matcher 成功计入。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHOD = "generic_mesh_fpfh_ransac_multiscale_icp_visible_surface_refinement"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    controller = ROOT / "src/vision_moveit_demo/vision_moveit_demo/cad_pick_place.py"
    # 对象名只能用于目录/场景登记，不能决定控制分支。
    for node in ast.walk(ast.parse(controller.read_text())):
        if isinstance(node, ast.If):
            constants = {item.value for item in ast.walk(node.test) if isinstance(item, ast.Constant) and isinstance(item.value, str)}
            assert not constants.intersection({"yellow_mug", "orange_rabbit", "magenta_block"}), "发现对象专属控制分支"
    records = []
    settings = []
    for path in args.episodes:
        episode = json.loads(path.read_text())
        assert episode["cad_model"]["geometry_type"] == "mesh"
        assert episode["pose_estimate"]["matching_method"] == METHOD
        assert episode["evaluation_only_success"] is True
        events = {event["name"]: event["details"] for event in episode["events"]}
        attached = events["grasp_attached"]
        assert attached["left_contacts"] > 0 and attached["right_contacts"] > 0
        assert attached["closing_axis_alignment"] >= 0.95
        assert attached["opposing_contact_pair"] == "passed"
        assert attached["required_object_geometries"]
        assert events["tray_verification"]["result"] == "passed"
        assert events["grasp_released_physical"]["qpos_reset"] == 0
        assert events["grasp_released_physical"]["qvel_reset"] == 0
        release = episode["physical_release"]
        assert release["qpos_reset_at_release"] is False and release["qvel_reset_at_release"] is False
        active = episode["active_observation"]
        settings.append((active["mode"], active["max_wrist_observations"]))
        assert active["grasp_roi"]["supporting_view_count"] >= 2
        records.append({"episode": str(path), "target": episode["target_object_id"], "success": True,
                        "left_contacts": attached["left_contacts"], "right_contacts": attached["right_contacts"],
                        "roi": active["grasp_roi"]})
    assert len({record["target"] for record in records}) >= 2, "验收必须涵盖不同 CAD"
    assert len(set(settings)) == 1, "两者必须使用相同主动观察配置"
    sources = ROOT / "src/vision_moveit_demo/vision_moveit_demo"
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(sources.glob("*.py"))}
    report = {"shared_matcher": METHOD, "shared_observation_configuration": list(settings[0]),
              "source_hashes_at_verification": hashes, "records": records}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": len(records), "targets": [record["target"] for record in records]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
