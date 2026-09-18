#include <memory>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

class MoveItTargetBridge : public rclcpp::Node {
 public:
  explicit MoveItTargetBridge(const rclcpp::NodeOptions& options)
      : Node("moveit_target_bridge", options) {
    execute_in_simulation_ = declare_parameter<bool>("execute_in_simulation", false);
    trajectory_publisher_ = create_publisher<moveit_msgs::msg::RobotTrajectory>(
        "/vision_moveit/planned_trajectory", 10);
    status_publisher_ = create_publisher<std_msgs::msg::String>("/vision_moveit/planning_status", 10);
    subscription_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/vision_moveit/target_pose", 10,
        std::bind(&MoveItTargetBridge::plan_target, this, std::placeholders::_1));
  }

  void initialise() {
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), "panda_arm");
    move_group_->setPoseReferenceFrame("panda_link0");
    move_group_->setPlanningTime(5.0);
    publish_status("规划桥已就绪，等待 /vision_moveit/target_pose");
  }

 private:
  void publish_status(const std::string& message) {
    std_msgs::msg::String status;
    status.data = message;
    status_publisher_->publish(status);
    RCLCPP_INFO(get_logger(), "%s", message.c_str());
  }

  void plan_target(const geometry_msgs::msg::PoseStamped::SharedPtr target) {
    if (!move_group_) {
      publish_status("规划桥尚未初始化，拒绝目标");
      return;
    }
    if (target->header.frame_id != "panda_link0" && target->header.frame_id != "base_link") {
      publish_status("目标坐标系错误，必须为 panda_link0 或 base_link");
      return;
    }
    move_group_->setPoseTarget(target->pose);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const auto result = move_group_->plan(plan);
    move_group_->clearPoseTargets();
    if (!result) {
      publish_status("MoveIt 规划失败");
      return;
    }
    trajectory_publisher_->publish(plan.trajectory);
    if (!execute_in_simulation_) {
      publish_status("MoveIt 规划成功，已发布关节轨迹（未执行）");
      return;
    }

    const auto execution_result = move_group_->execute(plan);
    if (!execution_result) {
      publish_status("MoveIt 规划成功，但模拟控制器执行失败");
      return;
    }
    publish_status("MoveIt 规划成功，模拟 Panda 已执行关节轨迹");
  }

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr subscription_;
  rclcpp::Publisher<moveit_msgs::msg::RobotTrajectory>::SharedPtr trajectory_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_publisher_;
  bool execute_in_simulation_{false};
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MoveItTargetBridge>(rclcpp::NodeOptions());
  node->initialise();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
