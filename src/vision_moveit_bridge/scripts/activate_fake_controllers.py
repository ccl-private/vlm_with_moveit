#!/usr/bin/env python3
"""等待 ros2_control 异步完成后，可靠激活 Panda 的三个模拟控制器。"""

import sys
import time

import rclpy
from builtin_interfaces.msg import Duration
from controller_manager_msgs.srv import (
    ConfigureController,
    ListControllers,
    LoadController,
    SwitchController,
)
from rclpy.node import Node


CONTROLLERS = (
    "joint_state_broadcaster",
    "panda_arm_controller",
    "panda_hand_controller",
)


class ControllerActivator(Node):
    def __init__(self):
        super().__init__("panda_controller_activator")
        self.list_client = self.create_client(
            ListControllers, "/controller_manager/list_controllers"
        )
        self.load_client = self.create_client(
            LoadController, "/controller_manager/load_controller"
        )
        self.configure_client = self.create_client(
            ConfigureController, "/controller_manager/configure_controller"
        )
        self.switch_client = self.create_client(
            SwitchController, "/controller_manager/switch_controller"
        )

    def wait_for_services(self):
        for client in (
            self.list_client,
            self.load_client,
            self.configure_client,
            self.switch_client,
        ):
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("等待 controller_manager 服务…")

    def call(self, client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if future.result() is None:
            raise RuntimeError("控制器管理器服务在 10 秒内未返回")
        return future.result()

    def states(self):
        response = self.call(self.list_client, ListControllers.Request())
        return {controller.name: controller.state for controller in response.controller}

    def wait_for_state(self, name, accepted, timeout=20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.states().get(name)
            if state in accepted:
                return state
            time.sleep(0.25)
        raise RuntimeError(f"控制器 {name} 未在 {timeout:g} 秒内进入 {accepted}")

    def activate(self):
        self.wait_for_services()
        for name in CONTROLLERS:
            if name not in self.states():
                request = LoadController.Request()
                request.name = name
                self.call(self.load_client, request)
                # Jazzy/RoboStack 的服务可能先返回 ok=false，随后才完成实际装载；因此以状态为准。
                self.wait_for_state(name, {"unconfigured", "inactive", "active"})

        for name in CONTROLLERS:
            state = self.states().get(name)
            if state == "unconfigured":
                request = ConfigureController.Request()
                request.name = name
                self.call(self.configure_client, request)
                self.wait_for_state(name, {"inactive", "active"})

        current = self.states()
        pending = [name for name in CONTROLLERS if current.get(name) != "active"]
        if pending:
            request = SwitchController.Request()
            request.activate_controllers = pending
            request.deactivate_controllers = []
            request.strictness = SwitchController.Request.STRICT
            request.activate_asap = True
            request.timeout = Duration(sec=10)
            self.call(self.switch_client, request)
            for name in pending:
                self.wait_for_state(name, {"active"})
        self.get_logger().info("Panda 模拟控制器均已激活")


def main():
    rclpy.init()
    node = ControllerActivator()
    try:
        node.activate()
    except Exception as error:
        node.get_logger().error(str(error))
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
