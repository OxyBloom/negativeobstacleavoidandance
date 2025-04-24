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
    lifecycle_nodes = ["map_saver_server"]
    map_file = os.path.join(
        get_package_share_directory("bumperbot_mapping"),  # Replace with your map package name
        "maps",
        "warehouse_map",
        "map.yaml"
    )

    nav2_config = os.path.join(
        get_package_share_directory("bot_nav2"),  # Replace with your navigation package name
        "config",
        "nav2_params.yaml"
    )

    # Nav2 bringup launch
    nav2 = IncludeLaunchDescription(
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
            "params_file": nav2_config,  # Add the parameter file
        }.items(),
    )

    nav2_lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_slam",
        output="screen",
        parameters=[
            {"node_names": lifecycle_nodes},
            {"use_sim_time": use_sim_time},
            {"autostart": True}
        ],
    )
    
    
    nav2_map_saver = Node(
        package="nav2_map_server",
        executable="map_saver_server",
        name="map_saver_server",
        output="screen",
        parameters=[
            {"save_map_timeout": 5.0},
            {"use_sim_time": use_sim_time},
            {"free_thresh_default", "0.196"},
            {"occupied_thresh_default", "0.65"},
        ],
    )


    return LaunchDescription([
        nav2,
        nav2_lifecycle_manager,
        nav2_map_saver,
    ])
