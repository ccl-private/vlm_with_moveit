"""VNC 阶段 1 预览器：全局 MuJoCo 画面叠加固定和腕部相机。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import queue
import threading
from pathlib import Path

import numpy as np

from .unified_scene import UnifiedPandaCupSimulation


def _resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.linspace(0, image.shape[0] - 1, height).astype(np.intp)
    cols = np.linspace(0, image.shape[1] - 1, width).astype(np.intp)
    return image[rows][:, cols]


def _depth_rgb(depth: np.ndarray) -> np.ndarray:
    """用于 VNC 观察的深度伪彩，不参与感知计算。"""
    normalized = np.clip((depth - 0.15) / 1.8, 0.0, 1.0)
    return np.stack(
        [255.0 * normalized, 255.0 * (1.0 - np.abs(2.0 * normalized - 1.0)), 255.0 * (1.0 - normalized)], axis=-1
    ).astype(np.uint8)


def _save_ppm(path: Path, rgb: np.ndarray) -> None:
    with path.open("wb") as output:
        output.write(f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode())
        output.write(rgb.tobytes())


class _Mp4Recorder:
    """通过 ffmpeg 将 RGB 帧编码为 MP4，不阻塞 MuJoCo 控制循环。"""

    def __init__(self, path: Path, width: int, height: int, fps: int = 30) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fps = fps
        self.wall_start: float | None = None
        self.frame_count = 0
        self.last_frame: bytes | None = None
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        self.pending_frames: queue.SimpleQueue = queue.SimpleQueue()
        self.recording_error: Exception | None = None
        self.worker = threading.Thread(target=self._encode_frames, daemon=True)
        self.worker.start()

    def _encode_frames(self) -> None:
        try:
            while True:
                item = self.pending_frames.get()
                if item is None:
                    break
                self._write_frame(*item)
        except Exception as error:
            self.recording_error = error

    def write(self, rgb: np.ndarray) -> None:
        if self.recording_error is not None:
            raise RuntimeError(f"视频编码失败：{self.path}") from self.recording_error
        frame = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
        self.pending_frames.put((frame, time.monotonic()))

    def _write_frame(self, frame: bytes, now: float) -> None:
        if self.process.stdin is None:
            raise RuntimeError(f"视频录制管道不可用：{self.path}")
        try:
            if self.wall_start is None:
                self.wall_start = now
            desired_count = int((now - self.wall_start) * self.fps) + 1
            if self.frame_count >= desired_count:
                self.last_frame = frame
                return
            # 记录实际墙钟节奏，保留规划等待与渲染停顿；不把低帧率执行
            # 压缩成看似更快的固定 30 FPS 视频。
            while self.last_frame is not None and self.frame_count < desired_count - 1:
                self.process.stdin.write(self.last_frame)
                self.frame_count += 1
            self.last_frame = frame
            self.process.stdin.write(self.last_frame)
            self.frame_count += 1
        except BrokenPipeError as error:
            raise RuntimeError(f"ffmpeg 无法写入视频：{self.path}") from error

    def close(self) -> None:
        self.pending_frames.put(None)
        self.worker.join()
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg 录制失败（退出码 {return_code}）：{self.path}")
        if self.recording_error is not None:
            raise RuntimeError(f"视频编码失败：{self.path}") from self.recording_error


def _save_stage1_snapshot(path: Path, simulation: UnifiedPandaCupSimulation) -> None:
    """保存与三路首帧同一时刻的标定和同步快照，供阶段 1 验收查看。"""
    frames = simulation.cameras()
    snapshot = simulation.synchronization_snapshot()
    payload = {
        "timestamp_s": snapshot.timestamp,
        "joint_positions_rad_or_m": snapshot.joint_positions,
        "planning_scene_objects_base": {
            name: {
                "position_base_m": value["position_base_m"].tolist(),
                "rotation_base": value["rotation_base"].tolist(),
            }
            for name, value in snapshot.planning_scene_objects.items()
        },
        "cameras": {
            name: {
                "timestamp_s": frame.timestamp,
                "intrinsic": frame.intrinsic.tolist(),
                "world_from_camera": frame.world_from_camera.tolist(),
                "base_from_camera": frame.base_from_camera.tolist(),
            }
            for name, frame in frames.items()
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


class VncOverlayViewer:
    """不依赖父项目的最小 GLFW/MuJoCo viewer，含两个实时相机叠加图。"""

    def __init__(
        self,
        simulation: UnifiedPandaCupSimulation,
        title: str = "MoveIt 阶段 1",
        video_dir: Path | None = None,
    ) -> None:
        import glfw
        import mujoco

        def report_glfw_error(code: int, description: object) -> None:
            text = description.decode("utf-8", errors="replace") if isinstance(description, bytes) else str(description)
            print(f"[阶段 1] GLFW 错误 {code}: {text}", flush=True)

        glfw.set_error_callback(report_glfw_error)
        print("[阶段 1] 正在初始化 GLFW…", flush=True)
        if not glfw.init():
            raise RuntimeError("无法初始化 GLFW；请从 VNC 桌面终端运行，并确认 DISPLAY 已设置。")
        self.glfw, self.mujoco, self.simulation = glfw, mujoco, simulation
        self.title = title
        # 服务器 VNC/VirtualGL 下以 1152×768 呈现全局视图；该分辨率足以观察
        # 夹爪、杯子和固定相机叠加，同时避免 1440×960 的交换缓冲拖慢控制回合。
        self.window = glfw.create_window(1152, 768, f"{title}：Panda、固定相机与腕部相机", None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("无法创建 MuJoCo VNC 窗口。")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        glfw.show_window(self.window)
        glfw.focus_window(self.window)
        self.wrist_window = glfw.create_window(480, 360, f"{title}：腕部 RGB-D 相机", None, None)
        if self.wrist_window is None:
            glfw.destroy_window(self.window)
            glfw.terminate()
            raise RuntimeError("无法创建腕部相机窗口。")
        glfw.make_context_current(self.wrist_window)
        glfw.swap_interval(0)
        glfw.set_window_pos(self.wrist_window, 24, 80)
        glfw.show_window(self.wrist_window)
        self.wrist_context = mujoco.MjrContext(simulation.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        glfw.make_context_current(self.window)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.camera.fixedcamid = mujoco.mj_name2id(
            simulation.model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_global"
        )
        self.option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(simulation.model, maxgeom=2000)
        self.context = mujoco.MjrContext(simulation.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.show_depth = False
        self._fixed_preview_frame = None
        self._fixed_preview_time = 0.0
        self._fixed_preview_depth = False
        self.editor_active = False
        self._editor_confirmed = False
        self._editor_selected_index = 0
        self._editor_message = "按 E 进入场景编辑模式"
        self.video_recorders: tuple[_Mp4Recorder, _Mp4Recorder] | None = None
        if video_dir is not None:
            self.video_recorders = (
                _Mp4Recorder(video_dir / "main.mp4", simulation.width, simulation.height),
                _Mp4Recorder(video_dir / "wrist.mp4", simulation.width, simulation.height),
            )
            print(f"[视频] 正在保存两路 MP4：{video_dir / 'main.mp4'}、{video_dir / 'wrist.mp4'}", flush=True)
        glfw.set_key_callback(self.window, self._on_key)
        print("[阶段 1] VNC 窗口已创建；正在渲染（D：深度，R：重置，E：编辑，Esc：退出）。", flush=True)

    def _on_key(self, _window, key, _scancode, action, _mods) -> None:
        if action != self.glfw.PRESS:
            return
        if self.editor_active and self._handle_editor_key(key, _mods):
            return
        if key == self.glfw.KEY_D:
            self.show_depth = not self.show_depth
        elif key == self.glfw.KEY_R:
            self.simulation.reset()
            self._editor_message = "场景已重置"
        elif key == self.glfw.KEY_E:
            self.editor_active = True
            self._editor_confirmed = False
            self._editor_message = "编辑模式：选择物体后用方向键移动，Enter 确认"
        elif key == self.glfw.KEY_ESCAPE:
            self.glfw.set_window_should_close(self.window, True)

    def _handle_editor_key(self, key: int, mods: int) -> bool:
        editable_objects = self.simulation.EDITABLE_OBJECTS
        if key == self.glfw.KEY_N:
            self._editor_selected_index = (self._editor_selected_index + 1) % len(editable_objects)
            self._editor_message = f"已切换到 {editable_objects[self._editor_selected_index]}"
            return True
        if key == self.glfw.KEY_ENTER:
            self._editor_confirmed = True
            self.editor_active = False
            self._editor_message = "编辑已确认；将从当前场景重新感知并规划"
            return True
        if key == self.glfw.KEY_E:
            self.editor_active = False
            self._editor_message = "已退出编辑模式；当前改动已保留"
            return True
        distance = 0.05 if mods & self.glfw.MOD_SHIFT else 0.01
        delta_by_key = {
            self.glfw.KEY_LEFT: (-distance, 0.0),
            self.glfw.KEY_RIGHT: (distance, 0.0),
            self.glfw.KEY_DOWN: (0.0, -distance),
            self.glfw.KEY_UP: (0.0, distance),
        }
        selected_name = editable_objects[self._editor_selected_index]
        if key in delta_by_key:
            _accepted, self._editor_message = self.simulation.move_editable_object(
                selected_name, delta_by_key[key]
            )
            return True
        if key in (self.glfw.KEY_J, self.glfw.KEY_L):
            yaw_delta = np.deg2rad(5.0 if key == self.glfw.KEY_J else -5.0)
            _accepted, self._editor_message = self.simulation.move_editable_object(selected_name, (0.0, 0.0), yaw_delta)
            return True
        return False

    def edit_before_task(self) -> bool:
        """阻塞在任务前的编辑状态；Enter 确认，关闭窗口则取消本回合。"""
        self.editor_active = True
        self._editor_confirmed = False
        self._editor_message = "编辑模式：N 切换物体，方向键移动，Shift 加速，J/L 旋转，Enter 开始任务"
        while not self._editor_confirmed:
            selected_name = self.simulation.EDITABLE_OBJECTS[self._editor_selected_index]
            state = f"Scene edit: {selected_name} | {self._editor_message}"
            if not self.render_once(state):
                return False
            # 编辑期间只刷新正运动学，不推进物理。否则物体会在用户还没确认时
            # 因重力/接触求解发生微小漂移，且下一步 CAD 看到的不是所编辑的位置。
            self.mujoco.mj_forward(self.simulation.model, self.simulation.data)
            time.sleep(1.0 / 30.0)
        return True

    def _draw_camera(self, image: np.ndarray, left: int, bottom: int, width: int, height: int, context) -> None:
        """将连续 RGB 图像上传到指定 viewport。"""
        resized = np.ascontiguousarray(np.flipud(_resize_nearest(image, height, width)))
        viewport = self.mujoco.MjrRect(left, bottom, width, height)
        self.mujoco.mjr_drawPixels(resized.ravel(), None, viewport, context)

    def _render_wrist_window(self, image: np.ndarray, state: str) -> None:
        """独立腕部窗口规避 Xvnc/VirtualGL 的同窗多 viewport 传输缺陷。"""
        self.glfw.make_context_current(self.wrist_window)
        width, height = self.glfw.get_framebuffer_size(self.wrist_window)
        viewport = self.mujoco.MjrRect(0, 0, width, height)
        self.mujoco.mjr_rectangle(viewport, 0.0, 0.0, 0.0, 1.0)
        self._draw_camera(image, 0, 0, width, height, self.wrist_context)
        self.mujoco.mjr_overlay(
            self.mujoco.mjtFont.mjFONT_NORMAL,
            self.mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            "Wrist RGB-D Camera",
            f"Mode: {state} | D: depth mode in main window | Esc: close",
            self.wrist_context,
        )
        self.glfw.swap_buffers(self.wrist_window)

    def render_once(self, task_state: str = "Idle") -> bool:
        """绘制一次当前仿真状态；供阶段 2 执行器复用，不推进物理时钟。"""
        if self.glfw.window_should_close(self.window) or self.glfw.window_should_close(self.wrist_window):
            return False
        # 复用感知链已创建的渲染器。VirtualGL 下若额外创建离屏 renderer，跨两个
        # GLFW 窗口的纹理上下文会变成全黑；此处只取两路所需 RGB（D 键才取深度）。
        self.glfw.make_context_current(self.window)
        now = time.monotonic()
        refresh_fixed = (self._fixed_preview_frame is None or now - self._fixed_preview_time >= 0.5
                         or self._fixed_preview_depth != self.show_depth)
        names = ("global", "wrist", "fixed") if refresh_fixed else ("global", "wrist")
        frames = self.simulation.cameras(names, include_depth=self.show_depth)
        if refresh_fixed:
            self._fixed_preview_frame = frames["fixed"]
            self._fixed_preview_time = now
            self._fixed_preview_depth = self.show_depth
        global_rgb = frames["global"].rgb if not self.show_depth else _depth_rgb(frames["global"].depth)
        fixed = _depth_rgb(self._fixed_preview_frame.depth) if self.show_depth else self._fixed_preview_frame.rgb
        wrist = _depth_rgb(frames["wrist"].depth) if self.show_depth else frames["wrist"].rgb
        if self.video_recorders is not None:
            main_video = global_rgb.copy()
            inset_width = main_video.shape[1] // 4
            inset_height = int(inset_width * main_video.shape[0] / main_video.shape[1])
            inset = _resize_nearest(fixed, inset_height, inset_width)
            main_video[16 : 16 + inset_height, main_video.shape[1] - inset_width - 16 : -16] = inset
            self.video_recorders[0].write(main_video)
            self.video_recorders[1].write(wrist)
        self.glfw.make_context_current(self.window)
        width, height = self.glfw.get_framebuffer_size(self.window)
        viewport = self.mujoco.MjrRect(0, 0, width, height)
        self.mujoco.mjv_updateScene(
            self.simulation.model, self.simulation.data, self.option, None, self.camera,
            self.mujoco.mjtCatBit.mjCAT_ALL.value, self.scene,
        )
        self.mujoco.mjr_render(viewport, self.scene, self.context)
        overlay_width = max(220, width // 4)
        overlay_height = max(165, int(overlay_width * 0.75))
        camera_mode = "Depth" if self.show_depth else "RGB"
        self._draw_camera(
            fixed, width - overlay_width - 16, height - overlay_height - 16, overlay_width, overlay_height, self.context
        )
        self.mujoco.mjr_overlay(
            self.mujoco.mjtFont.mjFONT_NORMAL,
            self.mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            "Panda Pick-and-Place",
            f"State: {task_state} | Fixed camera: top-right | Wrist: separate window\n"
            f"Camera: {camera_mode} | D: depth  R: reset  E: edit  Esc: stop",
            self.context,
        )
        self.glfw.swap_buffers(self.window)
        self._render_wrist_window(wrist, task_state)
        self.glfw.poll_events()
        return True

    def close(self) -> None:
        """释放 VNC 专用上下文；允许任务脚本在窗口关闭后干净退出。"""
        if self.video_recorders is not None:
            for recorder in self.video_recorders:
                recorder.close()
            self.video_recorders = None
        self.context.free()
        self.wrist_context.free()
        self.glfw.destroy_window(self.wrist_window)
        self.glfw.destroy_window(self.window)
        self.glfw.terminate()

    def run(self) -> None:
        while self.render_once("Idle / home"):
            if self.editor_active:
                self.mujoco.mj_forward(self.simulation.model, self.simulation.data)
            else:
                self.simulation.hold_home(steps=1)
            time.sleep(1.0 / 30.0)
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="统一 Panda 场景的 VNC 双相机预览")
    parser.add_argument("--root", type=Path, default=Path(os.environ["MOVEIT_EXPERIMENT_ROOT"]))
    parser.add_argument("--headless", action="store_true", help="不创建窗口，只写入三路首帧。")
    parser.add_argument("--video-dir", type=Path, help="保存 main.mp4 和 wrist.mp4 的目录。")
    arguments = parser.parse_args()
    mode = "离屏验收" if arguments.headless else "VNC 实时预览"
    print(f"[阶段 1] 正在加载统一场景（{mode}）…", flush=True)
    # GLFW/GLX 下先创建可见窗口，再初始化离屏 RGB-D 渲染器；某些 VNC 环境中
    # 反过来初始化会在 OpenGL 上下文装载阶段无提示阻塞。
    simulation = UnifiedPandaCupSimulation(arguments.root, create_renderers=arguments.headless)
    try:
        if arguments.headless:
            print("[阶段 1] 场景与三路 RGB-D 渲染器已就绪。", flush=True)
            output = arguments.root / "logs" / "unified_preview"
            output.mkdir(parents=True, exist_ok=True)
            for name, frame in simulation.cameras().items():
                _save_ppm(output / f"{name}.ppm", frame.rgb)
            _save_stage1_snapshot(output / "stage1_snapshot.json", simulation)
            print(f"已写入统一场景三路相机首帧：{output}")
            return
        viewer = VncOverlayViewer(simulation, video_dir=arguments.video_dir)
        print("[阶段 1] 正在初始化三路 RGB-D 渲染器…", flush=True)
        simulation.initialize_renderers()
        viewer.glfw.make_context_current(viewer.window)
        print("[阶段 1] 场景与三路 RGB-D 渲染器已就绪。", flush=True)
        viewer.run()
    finally:
        simulation.close()


if __name__ == "__main__":
    main()
