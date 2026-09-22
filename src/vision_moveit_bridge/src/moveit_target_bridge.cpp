#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
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
    scene_subscription_ = create_subscription<moveit_msgs::msg::PlanningScene>(
        "/vision_moveit/planning_scene_diff", 10,
        std::bind(&MoveItTargetBridge::apply_scene_diff, this, std::placeholders::_1));
    joint_state_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
        "/vision_moveit/sim_joint_states", 10,
        std::bind(&MoveItTargetBridge::save_sim_joint_state, this, std::placeholders::_1));
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
    // MoveGroupInterface::plan() 会等待 move_group action 的响应。不能在 ROS
    // 订阅回调线程中同步等待，否则该线程无法再处理 action 的反馈，形成死锁。
    // 任务客户端严格串行地发送目标，故每次目标在独立工作线程中规划即可。
    const auto target_copy = *target;
    std::thread([this, target_copy]() { plan_target_worker(target_copy); }).detach();
  }

  void plan_target_worker(const geometry_msgs::msg::PoseStamped& target) {
    if (!move_group_) {
      publish_status("规划桥尚未初始化，拒绝目标");
      return;
    }
    const bool position_only = target.header.frame_id == "panda_link0_position_only";
    if (target.header.frame_id != "panda_link0" && target.header.frame_id != "base_link" && !position_only) {
      publish_status("目标坐标系错误，必须为 panda_link0 或 base_link");
      return;
    }
    set_start_state_from_simulation();
    if (position_only) {
      move_group_->setPositionTarget(target.pose.position.x, target.pose.position.y, target.pose.position.z);
    } else {
      move_group_->setPoseTarget(target.pose);
    }
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const auto result = move_group_->plan(plan);
    move_group_->clearPoseTargets();
    move_group_->setStartStateToCurrentState();
    if (!result) {
      publish_status("MoveIt 规划失败");
      return;
    }
    // 透传请求时间戳，客户端据此拒绝后台规划线程迟到发布的旧轨迹。
    plan.trajectory.joint_trajectory.header.stamp = target.header.stamp;
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

  void apply_scene_diff(const moveit_msgs::msg::PlanningScene::SharedPtr scene) {
    if (!planning_scene_interface_.applyPlanningScene(*scene)) {
      publish_status("MoveIt 碰撞场景同步失败");
      return;
    }
    publish_status("MoveIt 碰撞场景已同步");
  }

  void save_sim_joint_state(const sensor_msgs::msg::JointState::SharedPtr joint_state) {
    latest_sim_joint_state_ = *joint_state;
  }

  void set_start_state_from_simulation() {
    if (!latest_sim_joint_state_) {
      publish_status("未收到 MuJoCo 关节状态，使用 MoveIt 当前状态规划");
      return;
    }
    moveit::core::RobotState start_state(move_group_->getRobotModel());
    start_state.setToDefaultValues();
    const auto& message = *latest_sim_joint_state_;
    for (size_t index = 0; index < message.name.size() && index < message.position.size(); ++index) {
      const auto& variable_names = start_state.getRobotModel()->getVariableNames();
      if (std::find(variable_names.begin(), variable_names.end(), message.name[index]) != variable_names.end()) {
        start_state.setVariablePosition(message.name[index], message.position[index]);
      }
    }
    start_state.update();
    move_group_->setStartState(start_state);
  }

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr subscription_;
  rclcpp::Subscription<moveit_msgs::msg::PlanningScene>::SharedPtr scene_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscription_;
  rclcpp::Publisher<moveit_msgs::msg::RobotTrajectory>::SharedPtr trajectory_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_publisher_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
  std::optional<sensor_msgs::msg::JointState> latest_sim_joint_state_;
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
