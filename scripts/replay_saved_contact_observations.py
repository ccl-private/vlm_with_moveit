#!/usr/bin/env python3
"""Replay saved wrist RGB-D with its recorded coarse seed; no motion is executed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/vision_moveit_demo'))
from vision_moveit_demo.cad_matching import ObjectCatalog,RuleVlmAdapter,GraspRoiMatcher,CadPoseEstimate,SegmentationResult
from vision_moveit_demo.unified_scene import UnifiedCameraFrame


def recorded_pose(raw):
    return CadPoseEstimate(raw['model_id'],np.array(raw['position_m']),np.array(raw['orientation_xyzw']),
        np.array(raw['transform_base_object']),raw['surface_residual_m'],raw['inlier_ratio'],
        raw['segmentation_confidence'],raw['geometry_confidence'],raw['axial_yaw_observable'],
        np.array(raw['cylinder_axis_base']),raw['point_count'],raw['matching_method'],raw['object_yaw_rad'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();records=[]
    for path in args.episodes:
        payload=json.loads(path.read_text());seed=recorded_pose(payload['active_observation']['coarse_pose'])
        model=ObjectCatalog(ROOT).find(RuleVlmAdapter().infer(payload['instruction']))
        observations=[];inputs=[]
        for view in payload['active_observation']['views']:
            if 'debug_observation' not in view:continue
            original=Path(view['debug_observation'])
            inputs.append(dict(path=str(original),sha256=hashlib.sha256(original.read_bytes()).hexdigest()))
            with np.load(original) as raw:
                mask=raw['mask'].astype(bool);rows,cols=np.nonzero(mask)
                frame=UnifiedCameraFrame('saved_wrist',raw['rgb'],raw['depth'],raw['intrinsic'],
                    raw['base_from_camera'],raw['base_from_camera'],0.)
                segmentation=SegmentationResult(mask,1.,int(mask.sum()),
                    (int(cols.min()),int(rows.min()),int(cols.max())+1,int(rows.max())+1),source='saved_original_mask')
                observations.append((segmentation,frame))
        result=GraspRoiMatcher().refine(model,model.grasp_templates[0],seed,observations)
        original_T=np.array(payload['pose_estimate']['transform_base_object']);T=result.pose.transform_base_object
        record=dict(episode=str(path),episode_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            inputs=inputs,accepted=True,roi=result.as_dict(),
            replay_position_difference_m=float(np.linalg.norm(T[:3,3]-original_T[:3,3])),
            replay_rotation_difference_deg=GraspRoiMatcher._rotation_angle_deg(T[:3,:3]@original_T[:3,:3].T))
        # Two copies of one original exposure must not invent the hidden face.
        try:GraspRoiMatcher().refine(model,model.grasp_templates[0],seed,[observations[0],observations[0]])
        except RuntimeError as error:record.update(single_exposure_duplicate_rejected=True,rejection=str(error))
        else:raise AssertionError('duplicated single-sided view supplied a false opposing pair')
        assert record['replay_position_difference_m']<1e-8 and record['replay_rotation_difference_deg']<1e-4
        records.append(record)
    report=dict(kind='offline original exposure replay; not a new physical trial',records=records,
        current_source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'src/vision_moveit_demo/vision_moveit_demo').glob('*.py'))})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(positive_replays=len(records),single_exposure_negatives=len(records),all_passed=True)))


if __name__=='__main__':main()
