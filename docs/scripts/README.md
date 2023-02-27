# Shell Scripts for Setting up Environment

This directory holds shell scripts to set the environment variables and other aliases to make working with Brevitas, FINN and Pynq-Z1 easier.

## Setup Environment Variables and Source

Instead of this file being run with `./setup_env_source.sh` we are going to use `source setup_env_source.sh`. This is so the updated environment variables are done within the current shell rather than another subshell.

This script sets up environment variables and loads Xilinx settings files, and then prints the values of the environment variables to the terminal. The environment variables that are set are `FINN_XILINX_PATH`, `FINN_XILINX_VERSION`, and `PLATFORM_REPO_PATHS`. These variables are commonly used in Xilinx development workflows.

The script assumes that Xilinx tools are installed in the default location (/opt/Xilinx), and that the version being used is 2022.2. If your installation is located in a different directory, you will need to modify the script accordingly.

The script will print the values of the environment variables to the terminal, so you can verify that they have been set correctly.

## Update Visudo 

This script adds environment variable settings to the sudoers file. The sudoers file is a configuration file used by the sudo command, which allows users to run commands with the privileges of another user (usually the root user).

The script first checks if it is being run as the root user. If not, it prints an error message and exits with an error code.

If the script is being run as root, it adds the environment variable settings to the sudoers file using the echo command. Specifically, it adds three lines that use the `env_keep` option to specify that the `FINN_XILINX_PATH`, `FINN_XILINX_VERSION`, and `PLATFORM_REPO_PATHS` environment variables should be preserved when the sudo command is used.

Finally, the script prints a message indicating that the environment variables have been added to the sudoers file.

To use the script, save it as a file and run it as the root user using the command sudo ./scriptname.sh (replacing "scriptname.sh" with the actual name of the script file). The environment variables will then be available when running commands with sudo.


