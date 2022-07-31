import os
from glob import glob
from setuptools import setup

package_name = "mix_net"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name), glob("launch/*.launch.py")),
        # data
        (os.path.join("lib", package_name, "config"), glob("mix_net/config/*.ini")),
        (
            os.path.join("lib", package_name, "data", "map"),
            glob("mix_net/data/map/*"),
        ),
        (
            os.path.join("lib", package_name, "data", "inference_model", "indy_net"),
            glob("mix_net/data/inference_model/indy_net/*"),
        ),
        (
            os.path.join("lib", package_name, "data", "inference_model", "mix_net"),
            glob("mix_net/data/inference_model/mix_net/*"),
        ),
        # src
        (os.path.join("lib", package_name, "src"), glob("mix_net/src/*.py")),
        # utils
        (os.path.join("lib", package_name, "utils"), glob("mix_net/utils/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    author="Phillip Karle",
    author_email="phillip.karle@tum.de",
    maintainer="Phillip Karle",
    maintainer_email="phillip.karle@tum.de",
    description="MixNet: Structured Deep Neural Motion Prediction of Opposing Vehicles for Autonomous Racing",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["mix_net_node = mix_net.mix_net_node:main"],
    },
)
