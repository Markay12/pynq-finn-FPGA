# Pynq Board Static IP Address Setup

You may need to set up a static IP address for your computer to connect to the Pynq board online.

You will need to edit the file

```
sudo vim /etc/network/interfaces
```

The file will usually have some data in it that looks like this.

![Interfaces 1](https://github.com/Markay12/pynq-finn-FPGA/blob/main/docs/setup/assets/Interfaces_1.png?raw=true)

When updating this we are going to add the static IP `192.168.2.1` to this file.

```
iface eth0 inet static
	address 192.168.2.1
	netmask 255.255.255.0
```

The final file should look like this. Now you have finished.

![Interfaces Final](https://github.com/Markay12/pynq-finn-FPGA/blob/main/docs/setup/assets/Interfaces_2.png?raw=true)
