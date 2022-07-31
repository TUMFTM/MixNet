# For more information, please refer to https://aka.ms/vscode-docker-python
ARG TAG_BASE_IMAGE=0.0.0
FROM tumiac/base:$TAG_BASE_IMAGE

# first copying only the requirements file
WORKDIR /dev_ws

# Update apt
RUN apt-get update

# Python requirements
COPY requirements.txt ./requirements.txt

# Install python packages
RUN apt-get update && pip3 install --upgrade pip \
    && pip3 install --no-cache-dir -r ./requirements.txt --ignore-installed PyYAML \
    && rm -rf ~/.cache/pip/*

# Copy repo into the container
COPY . /dev_ws/src/mod_prediction

# Resolve dependencies
RUN bash -c ". install/setup.bash && \
    rosdep init && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src --rosdistro $ROS_DISTRO -r -y"

# Set entrypoint by sourcing overlay workspace
RUN echo '#!/bin/bash\nset -e\n\n# setup ros environment\nsource "/opt/ros/$ROS_DISTRO/setup.bash"\n. \
    /dev_ws/install/setup.bash\nexec "$@"' > /ros_entrypoint.sh && \
    chmod +x /ros_entrypoint.sh

# Build and test workspace
RUN bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && \
            colcon build --paths /dev_ws/src/mod_prediction/mix_net"

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
