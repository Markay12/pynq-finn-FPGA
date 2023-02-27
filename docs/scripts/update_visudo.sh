#!/bin/bash

# check if the script is being run as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 
   exit 1
fi

# add the environment variable settings to the sudoers file
echo "Defaults	env_keep += \"FINN_XILINX_PATH\"" >> /etc/sudoers
echo "Defaults	env_keep += \"FINN_XILINX_VERSION\"" >> /etc/sudoers
echo "Defaults	env_keep += \"PLATFORM_REPO_PATHS\"" >> /etc/sudoers

echo "Environment variables added to /etc/sudoers"

