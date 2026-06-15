import os
import ast
import re
import subprocess

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterFile


def resolve_netvlad_python() -> str:
    def candidate_works(python_path: str) -> bool:
        try:
            result = subprocess.run(
                [python_path, '-c', 'import cv2, numpy, torch'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
        except Exception:
            return False
        return result.returncode == 0

    candidates = []

    configured = os.environ.get('F_VIGS_NETVLAD_PYTHON')
    if configured:
        candidates.append(os.path.expanduser(configured))

    candidates.append(os.path.expanduser('~/.pyenv/versions/hloc_env/bin/python3'))
    candidates.append('python3')

    for python_path in candidates:
        if candidate_works(python_path):
            return python_path

    return os.path.expanduser('~/.pyenv/versions/hloc_env/bin/python3')


def load_dataset_parameters(dataset_file: str) -> dict:
    def parse_scalar(raw_value: str):
        value = raw_value.strip()
        if value == '':
            return ''
        if value.startswith('[') and value.endswith(']'):
            return ast.literal_eval(value)
        if value[0] in ('"', "'") and value[-1] == value[0]:
            return value[1:-1]
        lowered = value.lower()
        if lowered == 'true':
            return True
        if lowered == 'false':
            return False
        if re.fullmatch(r'[-+]?\d+', value):
            return int(value)
        if re.fullmatch(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', value) or re.fullmatch(r'[-+]?\d+(?:[eE][-+]?\d+)', value):
            return float(value)
        return value

    def parse_dataset_file(file_path: str) -> dict:
        root: dict = {}
        stack: list[tuple[int, dict]] = [(-1, root)]

        with open(file_path, 'r', encoding='utf-8') as file_handle:
            for raw_line in file_handle:
                line = raw_line.split('#', 1)[0].rstrip()
                if not line.strip():
                    continue

                indent = len(line) - len(line.lstrip(' '))
                text = line.strip()

                while len(stack) > 1 and indent <= stack[-1][0]:
                    stack.pop()

                parent = stack[-1][1]

                if text.endswith(':'):
                    key = text[:-1].strip()
                    nested: dict = {}
                    parent[key] = nested
                    stack.append((indent, nested))
                    continue

                if ':' not in text:
                    continue

                key, raw_value = text.split(':', 1)
                parent[key.strip()] = parse_scalar(raw_value)

        return root

    with open(dataset_file, 'r', encoding='utf-8') as file_handle:
        try:
            yaml_module = __import__('yaml')
        except ImportError:
            yaml_module = None

        if yaml_module is not None:
            config = yaml_module.safe_load(file_handle) or {}
        else:
            file_handle.close()
            config = parse_dataset_file(dataset_file)

    node_config = config.get('gs_slam_node', {})
    parameters = node_config.get('ros__parameters', {})

    if not isinstance(parameters, dict):
        raise RuntimeError(f'Invalid dataset config format in {dataset_file}')

    return parameters


def static_tf_node(parent_frame: str,
                   child_frame: str,
                   translation: list,
                   rotation: list) -> Node:
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            "--x", str(translation[0]),
            "--y", str(translation[1]),
            "--z", str(translation[2]),
            "--qx", str(rotation[0]),
            "--qy", str(rotation[1]),
            "--qz", str(rotation[2]),
            "--qw", str(rotation[3]),
            "--frame-id", parent_frame,
            "--child-frame-id", child_frame,
        ],
    )


def launch_setup(context, *args, **kwargs):

    # Para descriptor NetVLAD
    pkg_share = FindPackageShare('f_vigs_slam').perform(context)
    server_script = os.path.join(pkg_share, 'scripts', 'netvlad_socket_server.py')
    netvlad_python = resolve_netvlad_python()

    print(f'[f_vigs_slam] NetVLAD server python: {netvlad_python}')
    print(f'[f_vigs_slam] NetVLAD server script: {server_script}')

    hloc_server = ExecuteProcess(
        cmd=[
            netvlad_python,
            server_script,
        ],
        output="screen"
    )

    dataset = LaunchConfiguration('dataset').perform(context)

    slam_base = ParameterFile(
        os.path.join(pkg_share, 'config', 'slam_base.yaml'),
        allow_substs=True
    )

    dataset_file_path = os.path.join(pkg_share, 'config', 'datasets', f'{dataset}.yaml')
    dataset_parameters = load_dataset_parameters(dataset_file_path)
    use_depth_registration = bool(dataset_parameters.get('use_depth_registration', False))
    depth_input_topic = dataset_parameters.get('depth_input_topic', dataset_parameters.get('depth_topic', ''))
    depth_output_topic = dataset_parameters.get('depth_topic', '')
    color_info_topic = dataset_parameters.get('camera_info_topic', '')
    depth_info_topic = dataset_parameters.get('depth_camera_info_topic', '')
    target_frame = dataset_parameters.get('depth_registration_target_frame', 'camera_color_optical_frame')
    source_frame = dataset_parameters.get('depth_registration_source_frame', 'camera_depth_optical_frame')
    publish_imu_camera_tf = bool(dataset_parameters.get('publish_imu_camera_tf', False))
    publish_world_to_map_tf = bool(dataset_parameters.get('publish_world_to_map_tf', False))

    imu_camera_parent_frame = dataset_parameters.get('imu_camera_parent_frame', 'xsens_imu_link')
    imu_camera_child_frame = dataset_parameters.get('imu_camera_child_frame', 'camera_link')
    imu_camera_translation = list(dataset_parameters.get('imu_camera_translation', [0.0, 0.0, 0.0]))
    imu_camera_rotation = list(dataset_parameters.get('imu_camera_rotation', [0.0, 0.0, 0.0, 1.0]))

    world_to_map_parent_frame = dataset_parameters.get('world_to_map_parent_frame', 'map')
    world_to_map_child_frame = dataset_parameters.get('world_to_map_child_frame', 'world')
    world_to_map_translation = list(dataset_parameters.get('world_to_map_translation', [0.0, 0.0, 0.0]))
    world_to_map_rotation = list(dataset_parameters.get('world_to_map_rotation', [0.0, 0.0, 0.0, 1.0]))

    dataset_file = ParameterFile(
        dataset_file_path,
        allow_substs=True
    )

    depth_topic_for_slam = depth_output_topic if use_depth_registration else dataset_parameters.get('depth_topic', depth_input_topic)

    nodes = [

        hloc_server,

        TimerAction(
            period=10.0,
            actions=[
                Node(
                    package='f_vigs_slam',
                    executable='gs_slam_node',
                    output='screen',
                    parameters=[
                        slam_base,
                        dataset_file,
                        {
                            'depth_topic': depth_topic_for_slam,
                        }
                    ],
                ),
            ],
        ),

        *( [
            static_tf_node(
                imu_camera_parent_frame,
                imu_camera_child_frame,
                imu_camera_translation,
                imu_camera_rotation,
            )
        ] if publish_imu_camera_tf else [] ),

        *( [
            static_tf_node(
                world_to_map_parent_frame,
                world_to_map_child_frame,
                world_to_map_translation,
                world_to_map_rotation,
            )
        ] if publish_world_to_map_tf else [] ),

        # Metrics node: computes per-odom ATE using GT topic from dataset params
        Node(
            package='slam_metrics',
            executable='slam_metrics_node',
            output='screen',
            parameters=[
                slam_base,
                dataset_file,
                {'use_sim_time': True}
            ],
        ),
    ]

    if use_depth_registration:
        nodes.insert(
            2,
            Node(
                package='cuda_depth_register',
                executable='cuda_depth_register_node',
                output='screen',
                parameters=[{
                    'use_sim_time': dataset_parameters.get('use_sim_time', True),
                    'depth_input_topic': depth_input_topic,
                    'depth_output_topic': depth_output_topic,
                    'color_info_topic': color_info_topic,
                    'depth_info_topic': depth_info_topic,
                    'target_frame': target_frame,
                    'source_frame': source_frame,
                }],
            ),
        )

    return nodes


def generate_launch_description():

    return LaunchDescription([
        DeclareLaunchArgument(
            'dataset',
            default_value='vigs'
        ),
        OpaqueFunction(function=launch_setup)
    ])
