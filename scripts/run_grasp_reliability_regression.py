#!/usr/bin/env python3
"""Fixed default-layout full-task trials with owned ROS/display processes and video."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def stop_owned(process):
    if process is None:return
    for sig,timeout in [(signal.SIGINT,3),(signal.SIGTERM,2),(signal.SIGKILL,2)]:
        try:os.killpg(process.pid,sig)
        except ProcessLookupError:pass
        try:process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:continue
        # Launch can exit before children; signal only its private group again.
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        return
    raise RuntimeError('owned process did not terminate')


def sources():
    paths=list((ROOT/'src/vision_moveit_demo/vision_moveit_demo').glob('*.py'))
    paths+=list((ROOT/'assets/object_models').glob('*'))
    paths+=list((ROOT/'assets/panda_mjcf').glob('*.xml'))
    paths+=[Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def main():
    import rclpy
    from moveit_msgs.srv import GetPositionIK
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--targets',nargs='+',choices=['mug','rabbit'],default=['rabbit','mug'])
    parser.add_argument('--repeats',type=int,default=10)
    parser.add_argument('--display',type=int,default=96)
    parser.add_argument('--observation-mode',choices=['on_path','sweep'],default='sweep')
    args=parser.parse_args()
    if not 1<=args.repeats<=10:raise ValueError('repeats must be 1..10')
    if os.environ.get('ROS_DOMAIN_ID') not in {'230','231','232'}:raise ValueError('use isolated domain 230..232')
    if args.output.exists():raise FileExistsError('new output folder required')
    if Path(f'/tmp/.X11-unix/X{args.display}').exists():raise RuntimeError('display already in use')
    args.output.mkdir(parents=True);args.output=args.output.resolve()
    report=dict(kind='fixed default-layout physical full-task trials',records=[],completed=False,
        predeclared_targets=args.targets,predeclared_repeats=args.repeats,source_sha256=sources(),
        observation_mode=args.observation_mode,controlled_paired_ab=False,planning_seed_fixed=False)
    def save():
        p=args.output/'summary.json.tmp';p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');p.replace(args.output/'summary.json')
    save();display=planner=task=None
    for rel in report['source_sha256']:
        copied=args.output/'source'/rel;copied.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/rel,copied)
    rclpy.init();node=rclpy.create_node('grasp_reliability_trials');client=node.create_client(GetPositionIK,'/compute_ik')
    try:
        for _ in range(10):rclpy.spin_once(node,timeout_sec=.1)
        if client.service_is_ready():raise RuntimeError('domain already contains a planner')
        with (args.output/'display.log').open('w') as log:
            display=subprocess.Popen(['/usr/bin/Xvnc',f':{args.display}','-geometry','1600x1000','-depth','24',
                '-localhost','-SecurityTypes','None','-rfbport',str(5900+args.display),'-nolisten','tcp'],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        until=time.monotonic()+8
        while not Path(f'/tmp/.X11-unix/X{args.display}').exists():
            if display.poll() is not None or time.monotonic()>until:raise RuntimeError('private display failed')
            time.sleep(.1)
        for target in args.targets:
            instruction='抓取橙色兔子耳朵并放到托盘' if target=='rabbit' else '抓取黄色马克杯把手并放到托盘'
            for repeat in range(args.repeats):
                if sources()!=report['source_sha256']:raise RuntimeError('source changed during trials')
                folder=args.output/f'{target}_{repeat:02d}';folder.mkdir()
                env=dict(os.environ,DISPLAY=f':{args.display}',MUJOCO_GL='glfw',
                    MOVEIT_EXPERIMENT_ROOT=str(ROOT),PYTHONPATH=str(ROOT/'src/vision_moveit_demo'),
                    ROS_LOG_DIR=str(folder/'ros'),ROS_HOME=str(folder/'ros_home'),
                    OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
                with (folder/'moveit.log').open('w') as log:
                    planner=subprocess.Popen(['ros2','launch','vision_moveit_bridge','headless_panda.launch.py'],
                        env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                until=time.monotonic()+30
                while not client.wait_for_service(timeout_sec=.25):
                    if planner.poll() is not None or time.monotonic()>until:raise RuntimeError('planner not ready')
                started=time.monotonic()
                with (folder/'task.log').open('w') as log:
                    task=subprocess.Popen(['vglrun','-d','egl0','--',sys.executable,'-m','vision_moveit_demo.cad_pick_place',
                        '--root',str(ROOT),'--instruction',instruction,'--force-matcher-type','mesh','--active-observation',
                        '--active-observation-mode',args.observation_mode,'--max-wrist-observations','6','--vnc',
                        '--episode-dir',str(folder/'episode'),'--video-dir',str(folder/'video'),
                        '--active-observation-debug-dir',str(folder/'observations')],env=env,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:code=task.wait(timeout=240)
                    except subprocess.TimeoutExpired:stop_owned(task);code=124
                stop_owned(task);task=None;stop_owned(planner);planner=None
                episode=folder/'episode/episode.json';payload=json.loads(episode.read_text()) if episode.exists() else {}
                trace_path=folder/'episode/runtime_trace.json'
                trace=json.loads(trace_path.read_text()) if trace_path.exists() else {}
                event_names={e['name'] for e in trace.get('events',[])}
                record=dict(target=target,repeat=repeat,exit_code=code,wall_s=time.monotonic()-started,
                    passed=code==0 and payload.get('evaluation_only_success') is True,
                    grasp_attached='grasp_attached' in event_names,error=trace.get('error'),
                    episode=str(episode),main_video=str(folder/'video/main.mp4'),wrist_video=str(folder/'video/wrist.mp4'))
                report['records'].append(record);save();print(json.dumps(record),flush=True)
        report['completed']=True
    finally:
        stop_owned(task);stop_owned(planner);stop_owned(display)
        node.destroy_node();rclpy.shutdown()
        report['source_unchanged']=sources()==report['source_sha256']
        report['all_passed']=report['completed'] and report['source_unchanged'] and all(r['passed'] for r in report['records'])
        save()
    if not report['all_passed']:raise SystemExit(1)


if __name__=='__main__':main()
