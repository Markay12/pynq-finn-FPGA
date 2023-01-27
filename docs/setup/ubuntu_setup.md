# Ubnutu Setup

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

You are now done and should have fully installed Docker Engine on your Ubuntu Bionic 18.04.\* Machine. For more information or troubleshooting with Docker follow this [link](https://docs.docker.com/engine/install/ubuntu/)

---

