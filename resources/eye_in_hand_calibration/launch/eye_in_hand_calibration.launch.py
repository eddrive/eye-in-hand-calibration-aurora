#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('eye_in_hand_calibration'),
            'config',
            'params.yaml'
        ]),
        description='Path to the configuration file'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time (true for rosbag playback)'
    )
    
    # Hand-eye calibrator node
    hand_eye_calibrator_node = Node(
        package='eye_in_hand_calibration',
        executable='hand_eye_calibrator_node',
        name='hand_eye_calibrator',
        parameters=[
            LaunchConfiguration('config_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}  # ← AGGIUNTO
        ],
        output='screen',
        emulate_tty=True,
        respawn=True,
        respawn_delay=2.0
    )
    
    return LaunchDescription([
        config_file_arg,
        use_sim_time_arg,  # ← AGGIUNTO
        LogInfo(msg=['Starting Eye-in-Hand Calibration with config: ', LaunchConfiguration('config_file')]),
        LogInfo(msg=['use_sim_time: ', LaunchConfiguration('use_sim_time')]),  # ← AGGIUNTO
        hand_eye_calibrator_node
    ])