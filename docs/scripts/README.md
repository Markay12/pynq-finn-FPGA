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

## SSH Key Generation and Installation Script

This shell script generates a new RSA key pair and installs the public key on a remote server for passwordless SSH authentication. It also creates a new directory `.ssh` at `/tmp/home_dir/` for storing the keys.

### Pre-requisites

- The script requires access to the `ssh-keygen` and `ssh` commands, which are usually pre-installed on most Unix-like systems.

- You need to have the necessary permissions to create a directory and write to files in the `/tmp/home_dir/` directory.

### Usage

1. Open the directory where `ssh_keygen_install.sh` is held.
2. `cat` the file to make sure the hostname and IP are correct. 
3. Save and close the file.
4. Make sure to set execution permissions. `chmod +x ssh_keygen_install.sh`
5. Run the script with `./ssh_keygen_install.ssh`

### Steps Performed by the Script

1. The script prompts the user to enter a filename to save the key pair. The default is `/tmp/home_dir/.ssh/id_rsa`.
2. The script prompts the user to enter a passphrase for the key pair. You can leave it empty for no passphrase.
3. The script generates the key pair using the ssh-keygen command.
4. The script creates the `.ssh` directory in the `/tmp/home_dir/` directory.
5. The script copies the public key to the remote server using the ssh command and adds it to the `authorized_keys` file in the `.ssh` directory of the remote user's home directory.
6. The script tests the key pair by logging into the remote server using the private key.

```shell
#!/bin/bash

# Set the IP address or hostname of the remote server
REMOTE_SERVER="10.206.148.244"

# Prompt the user for a filename to save the key pair
read -p "Enter file in which to save the key (/tmp/home_dir/.ssh/id_rsa): " KEY_FILE
KEY_FILE=${KEY_FILE:-/tmp/home_dir/.ssh/id_rsa}

# Prompt the user for a passphrase for the key pair
read -s -p "Enter passphrase (empty for no passphrase): " PASSPHRASE
echo
read -s -p "Enter same passphrase again: " PASSPHRASE2
echo

# Check that the two passphrases match
if [[ "$PASSPHRASE" != "$PASSPHRASE2" ]]; then
  echo "Passphrases do not match"
  exit 1
fi

# Generate the key pair
ssh-keygen -t rsa -b 2048 -f "$KEY_FILE" -N "$PASSPHRASE"

# Create the .ssh directory in /tmp/home_dir/
mkdir -p /tmp/home_dir/.ssh

# Copy the public key to the remote server
cat "${KEY_FILE}.pub" | ssh xilinx@${REMOTE_SERVER} "mkdir -p ~/.ssh && cat >>  ~/.ssh/authorized_keys"

# Test the key pair by logging into the remote server
ssh -i "$KEY_FILE" xilinx@${REMOTE_SERVER}
```

