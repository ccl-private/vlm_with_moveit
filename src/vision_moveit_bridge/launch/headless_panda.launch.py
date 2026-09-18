import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    start_rviz = LaunchConfiguration("start_rviz")
    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(
            file_path="config/panda.urdf.xacro",
            mappings={"ros2_control_hardware_type": "mock_components"},
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .planning_scene_monitor(
            publish_robot_description=True, publish_robot_description_semantic=True
        )
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .planning_pipelines(
            pipelines=["ompl", "chomp", "pilz_industrial_motion_planner", "stomp"]
        )
        .to_moveit_configs()
    )
    controllers = os.path.join(
        get_package_share_directory("moveit_resources_panda_moveit_config"),
        "config", "ros2_controllers.yaml",
    )
    rviz_config = os.path.join(
        get_package_share_directory("moveit_resources_panda_moveit_config"),
        "launch", "moveit.rviz",
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            "start_rviz",
            default_value="false",
            description="在有图形桌面的情况下启动 RViz。",
        ),
        Node(package="tf2_ros", executable="static_transform_publisher",
             arguments=["0", "0", "0", "0", "0", "0", "world", "panda_link0"]),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[moveit_config.robot_description]),
        Node(package="moveit_ros_move_group", executable="move_group", output="screen",
             parameters=[moveit_config.to_dict()]),
        Node(package="controller_manager", executable="ros2_control_node", output="screen",
             parameters=[controllers], remappings=[("/controller_manager/robot_description", "/robot_description")]),
        TimerAction(
            period=3.0,
            actions=[Node(
                package="vision_moveit_bridge",
                executable="activate_fake_controllers.py",
                output="screen",
            )],
        ),
        Node(package="vision_moveit_bridge", executable="moveit_target_bridge", output="screen",
             parameters=[moveit_config.to_dict(), {"execute_in_simulation": True}]),
        Node(
            package="rviz2",
            executable="rviz2",
            name="moveit_rviz",
            output="screen",
            arguments=["-d", rviz_config],
            parameters=[moveit_config.robot_description_kinematics],
            condition=IfCondition(start_rviz),
        ),
    ])
