# Full Setup Information

There are many requirements that are needed to setup your local machine to use Brevitas, FINN and the Pynq board together. These requirements are:

1. sudo (super user) access to the machine.
2. Docker Applicaton
	1. Docker will need to be set up as non-root for security reasons. A guide on how to do that will be included.
3. Xilinx Vitis and Vivado Installation
4. Setup of the `FINN_XILINX_PATH` and `FINN_XILINX_VERSION` environment variables.
5. Vitis and Vivado targeting the non-WebPack FPGA parts.
6. A PYNQ board connected to the ubuntu computer via ethernet connection or directly connected with no internet access.

---

# Table of Contents

1. [Installing Docker Engine](https://github.com/Markay12/pynq-finn-FPGA/blob/main/docs/setup/ubuntu_setup.md#installing-docker-engine)
	1. [Set Up Docker Engine Without Root](https://github.com/Markay12/pynq-finn-FPGA/blob/main/docs/setup/ubuntu_setup.md#setting-up-docker-engine-to-run-without-root)
2. [Install Vitis and Vivado](https://github.com/Markay12/pynq-finn-FPGA/tree/main/docs/setup#install-vitis-and-vivado-softare)
3. [Run Vitis and Vivado](https://github.com/Markay12/pynq-finn-FPGA/tree/main/docs/setup#launch-vivado)
4. [Set-up Environment Variables](https://github.com/Markay12/pynq-finn-FPGA/tree/main/docs/setup#set-up-environment-variables) 
5. [Clone FINN from Repository](https://github.com/Markay12/pynq-finn-FPGA/tree/main/docs/setup#clone-finn-compiler-from-repository)
6. [Verify FINN Installation](https://github.com/Markay12/pynq-finn-FPGA/tree/main/docs/setup#verify-installation-of-finn)

## Installing Docker Engine 

Note: All of these steps are performed on an Ubuntu system running Ubuntu Bionic 18.04.\* LTS. This is the version that should be used when recreating these steps as of January 2023.

Docker Engine is compatible with _x86 or amd64. armhf, arm64, s390x_ architectures.

Older versions of Docker should be uninstalled.

```Shell
$ sudo apt-get remove docker docker-engine docker.io containerd runc
```

The best way to set Docker up personally is by installing the repository. Steps to install by using the repository are found below.

### Set up the Repo

1. Set up the package index and then install packages to allow apt to use repo over HTTPS:

```Shell
$ sudo apt-get update

$ sudo apt-get install ca-certificates curl gnupg lsb-release
```

2. Add official Docker GPG key:

```Shell
$ sudo mkdir -p /etc/apt/keyrings

$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

3. Set up the repository:
```Shell
$ echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### Install Docker Engine

1. Update the apt package index:

```Shell
$ sudo apt-get update
```

2. Install the latest version of Docker Engine, containerd, and Docker Compose:

```Shell
$ sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

3. Verify that the Docker Engine installation is successful by running the `hello-world` image:

```Shell
$ sudo docker run hello-world
```

Explanation: This command downloads a test image and runs a container. When this finishes running a confirmation is made then docker is exited.

You are now done and should have fully installed Docker Engine on your Ubuntu Bionic 18.04.\* Machine. For more information or troubleshooting with Docker follow this [link](https://docs.docker.com/engine/install/ubuntu/).


## Setting Up Docker Engine to Run Without Root

Explanation for this Step: The docker daemon binds to a Unix socker, not a TCP port. By default it's the root user that owns the Unix socket, and other users can only access it using `sudo`. Therefore the docker daemon always runs as the root user.

Setting up docker this way allows us to run docker without the command sudo. Docker becomes its own group and users are added to that configuration. When the docker daemon starts, it creates a Unix socket accessible by members of the docker group. 

### Create the docker group and add your user profile

1. Create the `docker` group.

```Shell
$ sudo groupadd docker
```

2. Add your profile to the `docker` group.
```Shell
$ sudo usermod -aG docker $USER
```

Use $USER if you are on your own profile and the one that you want to add.

3. Logout and log back in so that your group membership is re-evaluated.

4. Verify your installation

```Shell
$ docker run hello-world
```

This command will download the test image as before but now doing it without the sudo command.

Docker is now set up to run without root access or the sudo command. For more information reference this [link](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

---

## Install Vitis and Vivado Softare

This installation is done assuming that you have access to a Xilinx license for the software.

1. Download the Vivado software from Xilinx. Found [here](https://www.xilinx.com/member/forms/download/xef.html?filename=Xilinx_Unified_2022.2_1014_8888_Lin64.bin).

2. This will download a `.bin` file. To intall using this extension do:

```Shell
$ chmod a+x <Directory/File-Name>
$ ./<Directory/File-Name>
```

This will then open the visual installer for Xilinx Vivado. Follow the steps by providing your login information and next steps. 

3. Select `Download and Install Now` not the image. 

4. Select the **Vitis** install.

5. Once this has been chosen, make sure that the Vivado install has also been selected.

![Vivado Selection](https://github.com/Markay12/pynq-finn-FPGA/blob/main/docs/setup/assets/VivadoSelection.png?raw=true)

6. Accept all terms.

7. Choose your installation directory.

8. Install Vitis and Vivado.

This may take a while as these two programs take up a lot of space. Have patience as well as a large storage space for these programs.

## Launch Vivado

To ensure Vivado installation we are going to open Xilinx Vivado. 

Xilinx has provided a script in the source folder of Vivado to set up the PATH for the current session. This is going to be used for now to make sure the installation works. We will change this setup later on to work with FINN.

To run the script navigate to the directory where Xilinx was installed.

From there run this command:

```Shell
source Vivado/2022.2/settings64.sh
source Vitis/2022.2/settings64.sh
```

Your PATH will now be set and you can run Vivado and Vitis with the command

`vivado&` or `vitis&` 


## Set-up Environment Variables

The next thing to do now that Vitis and Vivado is installed is to work on the FINN environment variables.

We need to set up `FINN_XILINX_PATH` AND `FINN_XILINX_VERSION` environment variables pointing respectively to the Xilinx tools installation directory.

In this case our finn path was installed to `/opt/Xilinx/` and the version we are using for Vivado and Vitis is `2022.2`. This will be important for the next step.

To set up environment variables we use the `export` command in Bash.

```Shell
export FINN_XILINX_PATH=/opt/Xilinx
export FINN_XILINX_VERSION=2022.2
export PLATFORM_REPO_PATHS=/opt/Xilinx/Vitis/2022.2/base_platforms
```

The environment variables have now been set to work with FINN.


## Clone FINN Compiler from Repository

Cloning FINN from the repository allows for the most updated version. Find which location you would like to install and use finn from. This is the pwd you want to be in for this clone command.

Once there, use this command to clone the FINN repo from Xilinx.

```Shell
git clone https://github.com/Xilinx/finn/
```

Then you will want to navigate inside the directory.

### Verify Installation of FINN

To verify your FINN installation you will use a docker quicktest and verify the output.

Run `./run-docker.sh quicktest` and ensure there are no errors.

After running this command a new message asked the PLATFORM\_REPO\_PATHS to be set for Vitis. This is done with the command `export PLATFORM_REPO_PATHS=/opt/Xilinx/Vitis/2022.2/base_platforms`

This final test may take some time. Once complete, make sure there are no errors. It is okay if some are skipped.

### Running Docker with Sudo

To run docker as sudo and still keep the environment variables, you will need to add Defaults to the visudo page.

To do this open visudo and add the defaults.

```Shell
sudo visudo
```

This will open visudo and add these lines.

```Shell
Defaults        env_keep += "FINN_XILINX_PATH"
Defaults        env_keep += "FINN_XILINX_VERSION"
Defaults        env_keep += "PLATFORM_REPO_PATHS"
``` 
