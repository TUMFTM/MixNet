from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Initialize launch parameters
    frequency = LaunchConfiguration("frequency", default=20.0)
    track = LaunchConfiguration("track", default="IMS")
    use_sim_time = LaunchConfiguration("use_sim_time", default=False)
    ci = LaunchConfiguration("ci", default=False)
    use_cuda = LaunchConfiguration("use_cuda", default=False)

    return LaunchDescription(
        [
            # Declare Arguments
            DeclareLaunchArgument(
                "frequency",
                default_value=frequency,
                description="Specify node frequency in Hz",
            ),
            DeclareLaunchArgument(
                "track",
                default_value=track,
                description="Specify track to use correct map, currently only 'IMS' supported",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value=use_sim_time,
                description="Set node to use sim time to replay ros2-bags",
            ),
            DeclareLaunchArgument(
                "ci",
                default_value=ci,
                description="trigger ci shut down",
            ),
            DeclareLaunchArgument(
                "use_cuda",
                default_value=use_cuda,
                description="Set boolean to use cuda",
            ),
            # Create Node
            Node(
                package="mix_net",
                executable="mix_net_node",
                name="MIX_NET",
                namespace="",
                parameters=[
                    {
                        "frequency": frequency,
                        "track": track,
                        "use_sim_time": use_sim_time,
                        "ci": ci,
                        "use_cuda": use_cuda,
                    }
                ],
                arguments=["--ros-args"],
            ),
        ]
    )
