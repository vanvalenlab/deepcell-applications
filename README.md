# deepcell-applications

A template of how to create and combine deep learning workflows using `deepcell-tf` and Docker.

## Build the image

```bash
docker build -t vanvalenlab/deepcell-applications .
```

## Run the image

For Docker API version >= 1.40:

```bash
export DATA_DIR=/path/to/data/dir
export MOUNT_DIR=/data
export IMAGE_FILE=example_image.tif
docker run -it --gpus 1 \
-v $DATA_DIR:$MOUNT_DIR \
vanvalenlab/deepcell-applications:latest $MOUNT_DIR/$IMAGE_FILE
```

For Docker API version < 1.40:

```bash
export DATA_DIR=/path/to/data/dir
export MOUNT_DIR=/data
export IMAGE_FILE=example_image.tif
NV_GPU=0 nvidia-docker run -it \
-v $DATA_DIR:$MOUNT_DIR \
vanvalenlab/deepcell-applications:latest $MOUNT_DIR/$IMAGE_FILE
```
