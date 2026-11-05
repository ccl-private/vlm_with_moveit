#!/usr/bin/env python3
"""Contact-focused observation geometry independent of ROS and simulator truth."""
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/vision_moveit_demo'))
from vision_moveit_demo.active_perception import generate_observation_views


class ContactViewTests(unittest.TestCase):
    def setUp(self):
        self.centers=np.array([[.26,-.388,.592],[.26,-.412,.592]])
        self.focus=self.centers.mean(axis=0)
        self.hand_camera=np.eye(4)
        self.hand_camera[:3,3]=[-.06,-.06,.12]
        self.views=generate_observation_views(self.focus,self.centers,
            np.array([[0.,1.,0.],[0.,-1.,0.]]),self.hand_camera,radius_m=.16,elevation_m=.16)

    def test_both_contact_sides_have_separate_observation_candidates(self):
        self.assertTrue(any(v.visible_contact_indices==(0,) for v in self.views))
        self.assertTrue(any(v.visible_contact_indices==(1,) for v in self.views))
        self.assertFalse(any(len(v.visible_contact_indices)==2 for v in self.views))

    def test_side_views_focus_at_contact_height_with_useful_depth(self):
        for view in self.views:
            self.assertGreater(view.camera_position_base_m[2]-self.focus[2],.15)
            self.assertGreater(np.linalg.norm(view.camera_position_base_m-self.focus),.20)

    def test_optical_axis_points_to_contact_center(self):
        from scipy.spatial.transform import Rotation
        for view in self.views:
            rotation=Rotation.from_quat(view.camera_orientation_base_xyzw).as_matrix()
            desired=self.focus-view.camera_position_base_m
            desired/=np.linalg.norm(desired)
            np.testing.assert_allclose(-rotation[:,2],desired,atol=1e-12)


if __name__=='__main__':unittest.main()
