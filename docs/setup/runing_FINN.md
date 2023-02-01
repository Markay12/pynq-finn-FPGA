# Running FINN in Docker

To run FINN in docker make sure that you are located in the directory where finn has been installed.

FINN runs inside a Docker container, and it comes with a script to easily build and launch the container. The run-docker.sh script can be run in multiple modes. The two main modes that are used include:

1. Jupyter Notebooks
2. Interactive Shell

Docker can also be launched with `build_dataflow`, however, FINN is more compiler infrastructure than compiler. This will allow command line entry for certain use cases.


## Jupyter Notebooks

To run the Jupyter notebook tutorials and get started with Jupyter notebooks use the command:

```Shell
bash ./run-docker.sh notebook
```

This should be done within the FINN directory.

## Interactive Shell

To run docker in an interactive shell with FINN you can use run-docker.sh without any additional arguments. Docker and all of its dependencies will open in a terminal to use for development.

```Shell
bash ./run-docker.sh
```


## Launch Build with `build_dataflow`

To launch with build\_dataflow use the command:

```Shell
bash ./run_docker.sh build_dataflow <path/to/dataflow_build_dir/>
bash ./run_docker.sh build_custom <path/to/custom_build_dir/>
```


