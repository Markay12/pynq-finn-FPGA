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

1. [Installing Docker Engine]()



## Installing Docker Engine 

Note: All of these steps are performed on an Ubuntu system running Ubuntu Bionic 18.04.\* LTS. This is the version that should be used when recreating these steps as of January 2023.

Docker Engine is compatible with _x86 or amd64. armhf, arm64, s390x_ architectures.

Older versions of Docker should be uninstalled.

```Bash
$ sudo apt-get remove docker docker-engine docker.io containerd runc
```


