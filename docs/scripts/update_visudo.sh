#!/bin/bash

# create a temporary file to store the sudoers changes
tmpfile=$(mktemp)

# add the lines to the temporary file
echo 'Defaults        env_keep += "FINN_XILINX_PATH"' >> $tmpfile
echo 'Defaults        env_keep += "FINN_XILINX_VERSION"' >> $tmpfile
echo 'Defaults        env_keep += "PLATFORM_REPO_PATHS"' >> $tmpfile

# check the syntax of the temporary file before copying it to sudoers
visudo -c -f $tmpfile

if [ $? -eq 0 ]; then
  # if syntax is OK, copy the temporary file to sudoers
  sudo cp $tmpfile /etc/sudoers.d/finn-env
  echo "sudoers file updated successfully"
else
  # if syntax is not OK, print an error message
  echo "Error: syntax check failed, sudoers file not updated"
fi

# remove the temporary file
rm $tmpfile

