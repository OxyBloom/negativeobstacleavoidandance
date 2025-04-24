import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():
    # Configurable launch arguments
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    map_file = os.path.join(
        get_package_share_directory("bot_nav2"),  # Replace with your map package name
        "maps", 
        "warehouse_map",# Replace with your map folder if different
        "map.yaml"  # Replace with your map file name if different
    )

    # Nav2 bringup launch
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("nav2_bringup"),
                "launch",
                "bringup_launch.py"
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "map": map_file,
        }.items(),
    )
    
    #rviz for navigation 
    # rviz_nav = Node(
    #     package="rviz2",
    #     executable="rviz2",
    #     arguments=["-d", os.path.join(
    #             get_package_share_directory("bumperbot_bringup"),
    #             "rviz",
    #             "new_small_house.rviz"
    #         )
    #     ],
    #     output="screen",
    #     parameters=[{"use_sim_time": True}],
    #     # condition=UnlessCondition(use_slam)
    # )

    return LaunchDescription([
        nav2_bringup,
        # rviz_nav
    ])
