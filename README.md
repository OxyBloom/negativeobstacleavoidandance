# Negative Obstacle Detection and Avoidance Research Work
This work demonstrates a negative obstacle detection and avoidance system for autonomous mobile navigation based on the integration of a YOLOv8 model, depth profile analysis and lidar fusion. 

Dataset can be downloaded [here](https://drive.google.com/drive/folders/1AqDTsmYQMycQP-AzVbiXb3G-W6KmRLzb?usp=sharing)

## :building_construction: Usage

This work can be used with an Ubuntu 22.04 machine running ROS 2 Humble.

### Prerequisites

* Install [Ubuntu 24.04](https://ubuntu.com/download/desktop) or [Ubuntu 22.04](https://releases.ubuntu.com/jammy/) on your PC or in a Virtual Machine.
* Install [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debians.html) if you are on Ubuntu 24.04. Otherwise, install [ROS 2 Humble](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debians.html) if you are on Ubuntu 22.04
* Install ROS 2 missing libraries. Some libraries that are used in this project are not in the standard ROS 2 package. Install them with:
* Install [VS Code](https://code.visualstudio.com/) and [Arduino IDE](https://docs.arduino.cc/software/ide-v2/tutorials/getting-started/ide-v2-downloading-and-installing/) on your PC in order to build and load the Arduino code on the device
* Install realsense gazebo plugin in not available: [Realsene Gazebo Plugin](https://github.com/pal-robotics/realsense_gazebo_plugin/tree/new_maintainer)
* Install Python and C++ addistional libraries
```sh
sudo apt-get update && sudo apt-get install -y \
     libserial-dev \
     python3-pip
```
* Other requirements:
```
ultralytics==8.0.196
numpy==1.24.1
```
```sh
pip install pyserial
```

### Installation

1. Create a Workspace
```sh
mkdir -p bumperbot/src
```

2. Put the code here
2. Clone this repo
```sh
cd bumperbot_ws/src
git clone https://github.com/OxyBloom/negative_obstacle_avoidandance.git
```

3. Install the dependencies
```sh
cd ..
rosdep install --from-paths src --ignore-src -i -y
```

4. Build the workspace
```sh
colcon build
```

5. Source the ROS Workspace (Use this command in a separate terminal from the one you used to build the workspace)
```sh
. install/setup.bash
```

#### :computer: Usage

1. Launch Simulated Environment in terminal 1 (Ensure your are in the directory)
```sh
. install/setup.bash
```
```sh
ros2 launch bumperbot_bringup simulated_robot.launch.py
```
2. Launch navigation system in terminal 2 (Ensure your are in the directory)
```sh
. install/setup.bash
```
```sh
ros2 launch bumperbot_localization nav.launch.py
```
3. Start negative obstacle detection in terminal 3 (Ensure your are in the directory)
```sh
. install/setup.bash
```
```sh
ros2 run bot_camera clean_yolo.py 
```

## :star2: Acknowledgements
* [Bumperbot](https://github.com/AntoBrandi/Bumper-Bot.git)
* [Turtlebot 3 Burger](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/)

## :link: Contact

David Esuga-Mopah - esugamopah@gmail.com
