#!/usr/bin/env python3
"""真实机器人模型下校验工作点、开度、遥测和视觉附着变换。"""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/vision_moveit_demo"))
from vision_moveit_demo.unified_scene import UnifiedPandaCupSimulation
from vision_moveit_demo.stage2_executor import MujocoTaskExecutor
from vision_moveit_demo.truth_baseline import MoveItTrajectoryClient
from vision_moveit_demo.cad_pick_place import _attached_pose_in_hand


class PlanningConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.sim = UnifiedPandaCupSimulation(ROOT, create_renderers=False)
        self.executor = MujocoTaskExecutor(self.sim)

    def tearDown(self):
        self.sim.close()

    def test_workpoint_matches_primary_pad_centers_at_different_arm_poses(self):
        model, data = self.sim.model, self.sim.data
        geometry = self.executor.gripper_geometry
        np.testing.assert_allclose(geometry.tcp_hand_m, [0, 0, .1029], atol=1e-9)
        for angle in (-.8, 0., .7):
            data.qpos[model.jnt_qposadr[model.joint("joint1").id]] = angle
            for index, value in ((1, .012), (2, .012)):
                data.qpos[model.jnt_qposadr[model.joint(f"finger_joint{index}").id]] = value
            mujoco.mj_forward(model, data)
            centers = []
            for name in ("left_finger", "right_finger"):
                body = model.body(name).id
                ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body
                       and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
                       and np.allclose(model.geom_size[g], [.0085, .004, .0085])]
                self.assertEqual(len(ids), 1)
                centers.append(data.geom_xpos[ids[0]])
            hand = model.body("hand").id
            tcp = data.xpos[hand] + data.xmat[hand].reshape(3, 3) @ geometry.tcp_hand_m
            np.testing.assert_allclose(tcp, np.mean(centers, axis=0), atol=1e-9)

    def test_annotated_width_is_reached_by_actual_servos(self):
        for width in (.022, .032):
            self.executor.set_gripper_width(width)
            state = self.executor.joint_positions()
            self.assertAlmostEqual(state["finger_joint1"], width / 2, delta=1e-5)
            self.assertAlmostEqual(state["finger_joint2"], width / 2, delta=1e-5)

    def test_impossible_width_is_rejected(self):
        for width in (-.001, .081, float("nan")):
            with self.assertRaises(ValueError):
                self.executor.set_gripper_width(width)

    def test_arm_bias_feedforward_removes_static_workpoint_sag(self):
        model, data = self.sim.model, self.sim.data
        hand = model.body("hand").id
        initial = data.xpos[hand].copy()
        for _ in range(1000):
            self.executor._advance_one_step()
        self.assertLess(np.linalg.norm(data.xpos[hand] - initial), .0001)
        # 外加力仅来自机械臂控制，不直接托举任何工件自由度。
        others = [i for i in range(model.nv) if i not in self.executor._arm_dofs]
        np.testing.assert_allclose(data.qfrc_applied[others], 0.0)

    def test_joint_publisher_retains_both_measured_fingers(self):
        import rclpy
        from builtin_interfaces.msg import Time
        sent = []
        client = SimpleNamespace(
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time())),
            joint_publisher=SimpleNamespace(publish=sent.append))
        state = self.executor.joint_positions()
        state.update(finger_joint1=.009, finger_joint2=.010)
        MoveItTrajectoryClient.publish_joint_state(client, state)
        joints = dict(zip(sent[0].name, sent[0].position))
        self.assertEqual(len(joints), 9)
        self.assertEqual(joints["panda_finger_joint1"], .009)
        self.assertEqual(joints["panda_finger_joint2"], .010)
        del state["finger_joint2"]
        with self.assertRaises(KeyError):
            MoveItTrajectoryClient.publish_joint_state(client, state)

    def test_attachment_roundtrip_uses_actual_hand_frame(self):
        hand = np.eye(4)
        hand[:3, :3] = Rotation.from_euler("xyz", [.2, -.3, .8]).as_matrix()
        hand[:3, 3] = [.4, -.2, .55]
        obj_rotation = Rotation.from_euler("z", -.5).as_matrix()
        obj_position = np.array([.43, -.17, .445])
        attached = _attached_pose_in_hand(obj_position, obj_rotation, hand)
        relative_position = np.array([attached.position.x, attached.position.y, attached.position.z])
        relative_rotation = Rotation.from_quat([attached.orientation.x, attached.orientation.y,
                                               attached.orientation.z, attached.orientation.w]).as_matrix()
        np.testing.assert_allclose(hand[:3, 3] + hand[:3, :3] @ relative_position, obj_position, atol=1e-12)
        np.testing.assert_allclose(hand[:3, :3] @ relative_rotation, obj_rotation, atol=1e-12)

    def test_support_lift_accepts_departure_and_rejects_downward_motion(self):
        model, data = self.sim.model, self.sim.data
        hand = model.body("hand").id
        relative = np.array([0., 0., .1])
        center = data.xpos[hand] + data.xmat[hand].reshape(3, 3) @ relative
        top = float(center[2] - .01 + .0002)
        names = [f"panda_joint{i}" for i in range(1, 8)]
        state = [self.executor.joint_positions()[f"joint{i}"] for i in range(1, 8)]
        upward = state.copy()
        upward[1] -= .04
        self.executor.validate_support_lift(names, [upward], relative, np.eye(3), "box", [.02]*3, top)
        downward = state.copy()
        downward[1] += .04
        for goal, table in ((downward, top), (upward, top + .003), (state, top)):
            with self.assertRaises(RuntimeError):
                self.executor.validate_support_lift(names, [goal], relative, np.eye(3), "box", [.02]*3, table)

    def test_support_contact_preserves_other_collision_rules_and_is_restored(self):
        from moveit_msgs.msg import AllowedCollisionEntry, AllowedCollisionMatrix
        from geometry_msgs.msg import Pose
        from vision_moveit_demo.cad_matching import ObjectCatalog
        scenes = []
        original = AllowedCollisionMatrix(entry_names=["hand", "link7"], entry_values=[
            AllowedCollisionEntry(enabled=[False, True]), AllowedCollisionEntry(enabled=[True, False])])
        client = SimpleNamespace(_published_world_object_ids=set(), _published_attached_object_ids=set(),
            _base_acm=original, _support_pair=None, scene_publisher=SimpleNamespace(publish=scenes.append),
            _trace=lambda *a: None)
        for allowed in (True, False):
            MoveItTrajectoryClient.publish_scene_positions(client, ObjectCatalog(ROOT).fixture_collision_positions(),
                grasp_target="yellow_mug", attached_object="yellow_mug", attached_pose=Pose(),
                allow_support_contact=allowed)
            matrix = scenes[-1].allowed_collision_matrix
            cup, table = [matrix.entry_names.index(n) for n in ("stage2_yellow_mug", "stage2_table")]
            self.assertEqual(matrix.entry_values[cup].enabled[table], allowed)
            self.assertTrue(matrix.entry_values[0].enabled[1])
            self.assertFalse(matrix.entry_values[table].enabled[0])
        self.assertEqual(original.entry_names, ["hand", "link7"])

    def test_collision_matrix_query_retries_a_late_first_response(self):
        from unittest.mock import patch
        from moveit_msgs.msg import AllowedCollisionMatrix
        matrix = AllowedCollisionMatrix(entry_names=["panda_hand"])
        cancelled = []
        first = SimpleNamespace(done=lambda: False, cancel=lambda: cancelled.append(True))
        second = SimpleNamespace(done=lambda: True,
            result=lambda: SimpleNamespace(scene=SimpleNamespace(allowed_collision_matrix=matrix)))
        pending = [first, second]
        service = SimpleNamespace(wait_for_service=lambda **k: True, call_async=lambda q: pending.pop(0))
        fake = SimpleNamespace(create_client=lambda *a: service, destroy_client=lambda c: None,
                               _trace=lambda *a: None)
        with patch("rclpy.spin_until_future_complete"):
            result = MoveItTrajectoryClient._read_collision_matrix(fake)
        self.assertEqual(result.entry_names, ["panda_hand"])
        self.assertEqual(cancelled, [True])
        self.assertEqual(pending, [])

    def test_execution_waits_for_matching_start_and_rejects_missing_attachment(self):
        import copy
        from geometry_msgs.msg import PoseStamped
        from moveit_msgs.msg import AttachedCollisionObject, RobotState, RobotTrajectory
        target = PoseStamped()
        target.header.stamp.sec = 42
        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.header.stamp = copy.deepcopy(target.header.stamp)
        start = RobotState()
        start.joint_state.header.stamp = copy.deepcopy(target.header.stamp)
        fake = SimpleNamespace(trajectories=[trajectory], planned_start_states=[],
                               _published_attached_object_ids={"stage2_yellow_mug"})
        self.assertIsNone(MoveItTrajectoryClient._matching_planning_result(fake, target))
        fake.planned_start_states = [start]
        with self.assertRaisesRegex(RuntimeError, "丢失手持物体"):
            MoveItTrajectoryClient._matching_planning_result(fake, target)
        attached = AttachedCollisionObject()
        attached.object.id = "stage2_yellow_mug"
        start.attached_collision_objects = [attached]
        self.assertIs(MoveItTrajectoryClient._matching_planning_result(fake, target), trajectory)
        start.joint_state.header.stamp.sec = 41
        self.assertIsNone(MoveItTrajectoryClient._matching_planning_result(fake, target))


if __name__ == "__main__":
    unittest.main()
