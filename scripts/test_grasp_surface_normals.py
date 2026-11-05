#!/usr/bin/env python3
"""Contact face identity regression: geometry, normal quality, signed views, fallback."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/vision_moveit_demo'))
from vision_moveit_demo.grasp_surface_geometry import SurfaceObservation, estimate_surface, nearest_surface, sample_contact_patch
from vision_moveit_demo.cad_matching import ObjectCatalog, RuleVlmAdapter, GraspRoiMatcher, CadPoseEstimate, ColorThresholdSegmenter
from vision_moveit_demo.unified_scene import UnifiedCameraFrame


def plane():
    x,y = np.meshgrid(np.linspace(-.02,.02,21), np.linspace(-.02,.02,21))
    return np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))


class SurfaceTests(unittest.TestCase):
    def test_planar_normals_and_camera_orientation(self):
        observation=estimate_surface(plane(),np.array([0,0,.1]))
        self.assertGreater(observation.valid.mean(),.9)
        self.assertTrue(np.all(observation.normals[observation.valid,2]>.999))

    def test_dense_close_range_plane_is_not_mistaken_for_insufficient_spread(self):
        observation=estimate_surface(plane()*.1,np.array([0,0,.1]))
        self.assertGreater(observation.valid.mean(),.9)

    def test_nearby_top_cannot_support_side(self):
        points=plane();observation=estimate_surface(points)
        normals=np.tile([0.,1.,0.],(len(points),1))
        distance,_,keep=nearest_surface(points,normals,observation,max_distance=.009,cosine=.9)
        self.assertFalse(keep.any())
        self.assertTrue(np.isinf(distance).all())

    def test_unoriented_plane_sign_does_not_change_result(self):
        points=plane();observation=estimate_surface(points)
        for sign in [-1,1]:
            _,_,keep=nearest_surface(points,np.tile([0,0,sign],(len(points),1)),
                observation,max_distance=.009,cosine=.9)
            self.assertGreater(keep.mean(),.9)

    def test_oriented_view_does_not_supply_invisible_reverse_face(self):
        points=plane();observation=estimate_surface(points,np.array([0,0,.1]))
        _,_,keep=nearest_surface(points,np.tile([0,0,-1],(len(points),1)),
            observation,max_distance=.009,cosine=.9)
        self.assertFalse(keep.any())

    def test_search_uses_nearest_compatible_point(self):
        observation=SurfaceObservation(np.array([[0.,0.,0.],[0.,.005,0.]]),
            np.array([[0.,0.,1.],[0.,1.,0.]]),np.ones(2,bool),np.array([0.,1.,1.]))
        distances,points,keep=nearest_surface(np.zeros((1,3)),np.array([[0.,1.,0.]]),
            observation,max_distance=.009,cosine=.9)
        self.assertTrue(keep[0])
        self.assertAlmostEqual(distances[0],.005)
        np.testing.assert_array_equal(points[0],observation.points[1])

    def test_unknown_normal_does_not_supply_support_at_zero_distance(self):
        observation=SurfaceObservation(np.zeros((1,3)),np.array([[0.,1.,0.]]),np.zeros(1,bool))
        _,_,keep=nearest_surface(np.zeros((1,3)),np.array([[0.,1.,0.]]),
            observation,max_distance=.009,cosine=.9)
        self.assertFalse(keep.any())

    def test_line_and_isolated_points_unknown(self):
        points=np.column_stack((np.linspace(-.01,.01,30),np.zeros(30),np.zeros(30)))
        self.assertFalse(estimate_surface(points).valid.any())
        self.assertFalse(estimate_surface(np.eye(3)).valid.any())

    def test_corner_fit_not_reported_as_flat_plane(self):
        grid=plane();vertical=grid[:,[0,2,1]]
        observation=estimate_surface(np.concatenate((grid,vertical)))
        intersection=np.linalg.norm(observation.points,axis=1)<.001
        self.assertFalse(observation.valid[intersection].any())

    def test_contact_samples_only_correct_faces_and_off_sharp_edges(self):
        for instruction in ['抓取橙色兔子耳朵并放到托盘','抓取黄色马克杯把手并放到托盘']:
            model=ObjectCatalog(ROOT).find(RuleVlmAdapter().infer(instruction))
            pair=model.grasp_templates[0].opposing_contact_pair
            for surface in [pair.positive_surface,pair.negative_surface]:
                points,normals=sample_contact_patch(model,surface,cosine=.94)
                self.assertGreaterEqual(len(points),16)
                self.assertTrue(np.all(normals@surface.normal_object>=.94))
                if model.model_id=='orange_rabbit_ears_v1':
                    self.assertTrue(np.all(abs(points[:,0]+.014)<.010-.0014))
                    self.assertTrue(np.all(abs(points[:,1]-surface.center_object_m[1])<1e-7))

    def test_raw_fallback_cannot_bypass_normal_gate(self):
        class NoRefinement(GraspRoiMatcher):
            def _refine_candidate(self,*args):return None
        model=ObjectCatalog(ROOT).find(RuleVlmAdapter().infer('抓取橙色兔子耳朵并放到托盘'))
        T=np.eye(4);T[:3,3]=[.28,-.40,.505]
        pose=CadPoseEstimate(model.model_id,T[:3,3],np.array([0,0,0,1.]),T,.002,1.,1.,.9,
            True,T[:3,2],1000,'generic_mesh_test',0.)
        top=plane()+[.266,-.40,.592]
        with self.assertRaisesRegex(RuntimeError,'抓取 ROI'):
            NoRefinement().refine_clouds(model,model.grasp_templates[0],pose,[top,top])

    def test_prepared_normals_cannot_drop_the_original_camera_orientation(self):
        points=plane()
        prepared=estimate_surface(points)  # Unoriented; invalid for signed RGB-D evidence.
        with self.assertRaisesRegex(ValueError,'prepared normals'):
            GraspRoiMatcher().refine_clouds(None,None,None,[points,points],
                camera_positions=[np.array([0.,0.,.1])]*2,
                surface_observations=[prepared,prepared])

    def test_local_seed_retains_shaded_face_without_table_or_red(self):
        rgb=np.full((30,40,3),[140,90,45],dtype=np.uint8)
        rgb[8:18,8:18]=[214,64,10]
        rgb[8:18,18:28]=[153,43,6]
        rgb[20:25,8:18]=[200,20,10]
        K=np.array([[100,0,20],[0,100,15],[0,0,1.]])
        camera=np.eye(4);camera[:3,3]=[.4,-.1,1.]
        frame=UnifiedCameraFrame('fixture',rgb,np.full((30,40),.5,np.float32),K,camera,camera,0.)
        intent=RuleVlmAdapter().infer('抓取橙色兔子耳朵并放到托盘')
        result=ColorThresholdSegmenter().segment(rgb,intent,frame=frame,
            expected_position_base_m=np.array([.4,-.1,.5]),lateral_radius_m=.2,z_min_m=.49,z_max_m=.51)
        self.assertEqual(result.pixel_count,200)
        self.assertFalse(result.mask[0,0])
        self.assertFalse(result.mask[22,12])
        self.assertTrue(result.mask[10,22])

    def test_global_color_gate_is_not_relaxed(self):
        rgb=np.full((30,40,3),[153,43,6],dtype=np.uint8)
        intent=RuleVlmAdapter().infer('抓取橙色兔子耳朵并放到托盘')
        with self.assertRaisesRegex(RuntimeError,'分割像素不足'):
            ColorThresholdSegmenter().segment(rgb,intent)


if __name__=='__main__':unittest.main()
