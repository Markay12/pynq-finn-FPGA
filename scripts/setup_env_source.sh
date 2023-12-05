#!/bin/bash

# set environment variables
export FINN_XILINX_PATH=/opt/Xilinx
export FINN_XILINX_VERSION=2022.2
export PLATFORM_REPO_PATHS=/opt/Xilinx/Vitis/2022.2/base_platforms

# load Xilinx settings files
source /opt/Xilinx/Vivado/2022.2/settings64.sh
source /opt/Xilinx/Vitis/2022.2/settings64.sh

# print the environment variables to the terminal
echo "FINN_XILINX_PATH is set to: $FINN_XILINX_PATH"
echo "FINN_XILINX_VERSION is set to: $FINN_XILINX_VERSION"
echo "PLATFORM_REPO_PATHS is set to: $PLATFORM_REPO_PATHS"
