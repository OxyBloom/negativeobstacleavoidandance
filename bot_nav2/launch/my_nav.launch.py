import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():
    # Configurable launch arguments
    use_sim_time = LaunchConfiguration("use_sim_time", default="True")

    # Nav2 bringup launch
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("bot_nav2"),
                "launch",
                "nav.launch.py"
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
        }.items(),
    )
    
    #localization bringup launch
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("bot_nav2"),
                "launch",
                "localizatin.launch.py"
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
        }.items(),
    )
    
    

    
    # rviz for navigation 
    rviz_nav = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", os.path.join(
                get_package_share_directory("bot_nav2"),
                "rviz",
                "nav.param.yaml"
            )
        ],
        output="screen",
        parameters=[{"use_sim_time": True}],
       
    )

    return LaunchDescription([
        nav2,
        rviz_nav
    ])