from setuptools import setup

package_name = "scara_demo"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SCARA project",
    maintainer_email="user@example.com",
    description="SCARA kinematics / trajectory / pick-place demo for ROS2 + Gazebo",
    license="MIT",
    entry_points={
        "console_scripts": [
            "pick_place = scara_demo.pick_place_node:main",
            "move_to = scara_demo.move_to:main",
            "check_tf = scara_demo.check_tf:main",
            "capture = scara_demo.capture:main",
            "watch_object = scara_demo.watch_object:main",
            "moveit_plan = scara_demo.moveit_plan:main",
            "vision_pick = scara_demo.vision_pick:main",
        ],
    },
)
