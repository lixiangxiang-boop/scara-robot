// Gazebo Classic 力控夹爪插件
// ---------------------------------------------------------------------------
// 为什么需要它：gazebo_ros2_control 把 position/velocity 命令实现为"运动学置位"，
// 指关节直接穿过物体，没有夹持力。这里给两指加一个**软位置伺服**（PD 力控）：
//     f = clamp(kp*(q_target - q) - kd*dq, ±fmax)
// 手指被物体挡住时停在接触位置，PD 力就是夹持力（可调 kp/fmax），从而能做真实摩擦抓取。
// 指令：std_msgs/Float64（0.012 = 张开，0.0 = 闭合），话题默认 /gripper/target。
// ---------------------------------------------------------------------------
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/Joint.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo_ros/node.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>

namespace scara_gripper
{
class GripperForcePlugin : public gazebo::ModelPlugin
{
public:
  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = model;
    ros_node_ = gazebo_ros::Node::Get(sdf);

    if (sdf->HasElement("kp")) kp_ = sdf->Get<double>("kp");
    if (sdf->HasElement("kd")) kd_ = sdf->Get<double>("kd");
    if (sdf->HasElement("fmax")) fmax_ = sdf->Get<double>("fmax");
    if (sdf->HasElement("topic")) topic_ = sdf->Get<std::string>("topic");

    // 读取所有 <joint> 子元素
    if (sdf->HasElement("joint")) {
      sdf::ElementPtr e = sdf->GetElement("joint");
      while (e) {
        const std::string name = e->Get<std::string>();
        auto joint = model_->GetJoint(name);
        if (joint) {
          joints_.push_back(joint);
        } else {
          RCLCPP_WARN(ros_node_->get_logger(), "找不到关节：%s", name.c_str());
        }
        e = e->GetNextElement("joint");
      }
    }

    sub_ = ros_node_->create_subscription<std_msgs::msg::Float64>(
      topic_, 10,
      [this](const std_msgs::msg::Float64::SharedPtr msg) { target_ = msg->data; });

    update_conn_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      [this](const gazebo::common::UpdateInfo &) { OnUpdate(); });

    RCLCPP_INFO(ros_node_->get_logger(),
                "力控夹爪已加载：%zu 指，kp=%.0f kd=%.0f fmax=%.1f N，话题 %s",
                joints_.size(), kp_, kd_, fmax_, topic_.c_str());
  }

private:
  void OnUpdate()
  {
    // 启动前 10 拍不施加力：等模型与求解器完全就绪（否则首拍大加速度会让 ODE 发散）
    if (updates_++ < 10) return;
    for (auto & joint : joints_) {
      const double q = joint->Position(0);
      const double dq = joint->GetVelocity(0);
      double f = kp_ * (target_ - q) - kd_ * dq;
      f = std::max(-fmax_, std::min(fmax_, f));
      joint->SetForce(0, f);
    }
  }

  gazebo::physics::ModelPtr model_;
  std::vector<gazebo::physics::JointPtr> joints_;
  gazebo_ros::Node::SharedPtr ros_node_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr sub_;
  gazebo::event::ConnectionPtr update_conn_;
  double kp_{800.0};
  double kd_{20.0};
  double fmax_{10.0};
  double target_{0.0};   // 与 URDF 的初始指位置一致，启动时力≈0
  std::string topic_{"/gripper/target"};
  int updates_{0};
};

GZ_REGISTER_MODEL_PLUGIN(GripperForcePlugin)
}  // namespace scara_gripper
