#!/usr/bin/env python3
"""Apply a frozen pre-task edit, then run the unchanged grasp controller."""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def read_manifest(path):
    manifest=json.loads(path.read_text())
    payload={k:v for k,v in manifest.items() if k!='sha256'}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    if digest!=manifest['sha256']:raise ValueError('frozen manifest hash mismatch')
    ids=[s['id'] for s in manifest['scenarios']]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate scenario id')
    for s in manifest['scenarios']:
        if (len(s['delta_xy_m'])!=2
                or not all(math.isfinite(x) and abs(x)<=.07+1e-9 for x in s['delta_xy_m'])
                or not math.isfinite(s['yaw_delta_rad']) or abs(s['yaw_delta_rad'])>math.pi/3+1e-9):
            raise ValueError('scenario perturbation outside frozen range')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--scenario',required=True)
    parser.add_argument('--episode-dir',type=Path,required=True)
    parser.add_argument('--video-dir',type=Path,required=True)
    parser.add_argument('--active-observation-debug-dir',type=Path,required=True)
    parser.add_argument('--observation-mode',choices=['on_path','sweep'],default='sweep')
    args=parser.parse_args();manifest=read_manifest(args.manifest)
    fixture=next(s for s in manifest['scenarios'] if s['id']==args.scenario)
    sys.path.insert(0,str(ROOT/'src/vision_moveit_demo'))
    os.environ['MOVEIT_EXPERIMENT_ROOT']=str(ROOT)
    os.environ['MUJOCO_GL']='glfw'
    task=importlib.import_module('vision_moveit_demo.cad_pick_place')
    original=task.UnifiedPandaCupSimulation
    edit={}

    class ScenarioSimulation(original):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            if fixture['random']:
                accepted,reason=self.move_editable_object(fixture['target'],tuple(fixture['delta_xy_m']),fixture['yaw_delta_rad'])
                edit.update(accepted=accepted,reason=reason)
                if not accepted:
                    self.close()
                    raise RuntimeError('frozen fixture edit rejected: '+reason)

    # Only scene construction is replaced. The pose still comes from fresh RGB-D.
    task.UnifiedPandaCupSimulation=ScenarioSimulation
    sys.argv=['cad_pick_place','--root',str(ROOT),'--instruction',fixture['instruction'],
        '--force-matcher-type','mesh','--active-observation','--active-observation-mode',args.observation_mode,
        '--max-wrist-observations','6','--vnc','--episode-dir',str(args.episode_dir),
        '--video-dir',str(args.video_dir),'--active-observation-debug-dir',str(args.active_observation_debug_dir)]
    args.episode_dir.mkdir(parents=True,exist_ok=True);started=time.monotonic();error=None
    try:task.main()
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc))
        raise
    finally:
        episode=args.episode_dir/'episode.json'
        payload=json.loads(episode.read_text()) if episode.exists() else {}
        record=dict(scenario_id=fixture['id'],manifest_sha256=manifest['sha256'],fixture=fixture,
            scene_edit=edit,source_root=str(ROOT),wall_duration_s=time.monotonic()-started,
            exception=error,episode_written=episode.exists(),evaluation_only_success=payload.get('evaluation_only_success') is True)
        (args.episode_dir/'scenario_result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
