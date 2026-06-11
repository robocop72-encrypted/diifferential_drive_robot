import os
import xacro
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

ROBOTS = [
    {'name': 'robot_1', 'x':  1.0, 'y':  1.0, 'yaw': 0.0},
    {'name': 'robot_2', 'x':  3.0, 'y':  1.0, 'yaw': 0.0},
]

WORKSPACE = '/workspaces/differential_drive_robot'
XACRO_PATH = os.path.join(WORKSPACE, 'multi_story_car.urdf.xacro')

def generate_launch_description():
    world_arg = DeclareLaunchArgument('world', default_value='maze_arena.world')
    world = LaunchConfiguration('world')

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('gazebo_ros'), 'launch', 'gazebo.launch.py'])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([WORKSPACE, 'arena', world]),
            'verbose': 'false',
        }.items()
    )

    robot_groups = []
    for i, robot in enumerate(ROBOTS):
        rname = robot['name']
        rx, ry, ryaw = str(robot['x']), str(robot['y']), str(robot['yaw'])

        # process xacro with robot_name argument
        robot_desc = xacro.process_file(
            XACRO_PATH,
            mappings={'robot_name': rname}
        ).toxml()

        # write processed urdf to /tmp for spawner
        urdf_tmp = f'/tmp/{rname}.urdf'
        with open(urdf_tmp, 'w') as f:
            f.write(robot_desc)

        spawn = TimerAction(period=3.0 + i * 2.0, actions=[
            Node(
                package='gazebo_ros', executable='spawn_entity.py',
                name=f'spawn_{rname}',
                arguments=['-entity', rname, '-file', urdf_tmp,
                           '-robot_namespace', rname,
                           '-x', rx, '-y', ry, '-z', '0.12',
                           '-R', '0', '-P', '0', '-Y', ryaw],
                output='screen'
            )
        ])

        rsp = Node(
            package='robot_state_publisher', executable='robot_state_publisher',
            name='robot_state_publisher', namespace=rname,
            parameters=[{'robot_description': robot_desc, 'use_sim_time': True}],
            output='screen'
        )

        fusion = TimerAction(period=5.0 + i * 2.0, actions=[
            Node(package='swarm_navigation', executable='sensor_fusion',
                 name=f'sensor_fusion_{rname}',
                 parameters=[{'robot_name': rname, 'use_sim_time': True}],
                 output='screen'),
            Node(package='swarm_navigation', executable='obstacle_classifier',
                 name=f'obstacle_classifier_{rname}',
                 parameters=[{'robot_name': rname, 'use_sim_time': True}],
                 output='screen'),
            Node(package='slam_toolbox', executable='async_slam_toolbox_node',
                 name=f'slam_toolbox_{rname}',
                 parameters=[
                     os.path.join(WORKSPACE, 'src', 'swarm_navigation', 'config', 'slam_config.yaml'),
                     {'use_sim_time': True}
                 ],
                 remappings=[
                     ('/scan', f'/{rname}/fused_scan'),
                     ('/odom', f'/{rname}/odom'),
                     ('/map',  f'/{rname}/map'),
                 ],
                 output='screen'),
        ])

        robot_groups += [spawn, rsp, fusion]

    rviz = TimerAction(period=8.0, actions=[
        Node(package='rviz2', executable='rviz2', name='rviz2',
             arguments=['-d', os.path.join(WORKSPACE, 'config', 'rviz_mapping.rviz')],
             parameters=[{'use_sim_time': True}], output='screen')
    ])

    return LaunchDescription([world_arg, gazebo_launch, *robot_groups, rviz])
