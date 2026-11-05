#!/usr/bin/env python3
"""Replay original per-view wrist evidence; no new exposure or motion is claimed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/vision_moveit_demo'))
from vision_moveit_demo.cad_matching import (ObjectCatalog,RuleVlmAdapter,GraspRoiMatcher,
    CadPoseEstimate,SegmentationResult)
from vision_moveit_demo.unified_scene import UnifiedCameraFrame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    old=json.loads((ROOT/'logs/reactive_foundation/legacy_default_rabbit_failure_replay_20261104.json').read_text())
    seed=old['reconstructed_seed'];T=np.array(seed['transform_base_object'])
    model=ObjectCatalog(ROOT).find(RuleVlmAdapter().infer('抓取橙色兔子耳朵并放到托盘'))
    pose=CadPoseEstimate(model.model_id,T[:3,3],np.array(seed['orientation_xyzw']),T,
        seed['surface_residual_m'],seed['inlier_ratio'],seed['segmentation_confidence'],
        seed['geometry_confidence'],seed['axial_yaw_observable'],T[:3,2],seed['point_count'],
        seed['matching_method'],seed['object_yaw_rad'])
    report=dict(kind='offline original per-view replay; not physical grasp success',records=[])
    for historical in old['records']:
        observations=[]
        for item in historical['original_wrist']:
            path=Path(item['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
            with np.load(path) as raw:
                frame=UnifiedCameraFrame('wrist',raw['rgb'],raw['depth'],raw['intrinsic'],
                    raw['base_from_camera'],raw['base_from_camera'],0.)
                mask=raw['mask'].astype(bool)
                rows,columns=np.nonzero(mask)
                segmentation=SegmentationResult(mask,1.,int(mask.sum()),
                    (int(columns.min()),int(rows.min()),int(columns.max())+1,int(rows.max())+1),
                    source='original_saved_wrist_mask')
                observations.append((segmentation,frame))
        started=time.monotonic();record=dict(scenario=historical['scenario'],repeat=historical['repeat'],
            historical_success=historical['historical_success'],historical_wrong_direction=historical['winning_refinement_returned_none'])
        try:
            result=GraspRoiMatcher().refine(model,model.grasp_templates[0],pose,observations)
            axis=result.pose.transform_base_object[:3,:3]@model.grasp_templates[0].jaw_closing_axis_object
            record.update(accepted=True,closing_axis_error_deg=float(np.degrees(np.arccos(np.clip(abs(axis[1]),0,1)))),
                roi=result.as_dict(),diagnostics=result.pose.registration_diagnostics)
        except RuntimeError as error:record.update(accepted=False,error=str(error))
        record['wall_s']=time.monotonic()-started;report['records'].append(record)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k not in ['roi','diagnostics']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
