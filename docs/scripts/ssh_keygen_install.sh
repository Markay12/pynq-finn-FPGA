#!/bin/bash

# Set the IP address or hostname of the remote server
REMOTE_SERVER="10.206.149.0"

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

