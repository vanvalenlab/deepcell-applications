# deepcell-applications

A template of how to create and combine deep learning workflows using `deepcell-tf` and Docker.

## Run the Python script

An example Python script `main.py` is provided as an example deepcell.application workflow.
It process Nuclear and Membrane stain images and saves an instance segmentation mask of either the whole cell or the nucleus.

```bash
export DATA_DIR=/path/to/data/dir
export MOUNT_DIR=/data
export NUCLEAR_FILE=example_nuclear_image.tif
export MEMBRANE_FILE=example_membrane_image.tif
python main.py \
  --nuclear-image $MOUNT_DIR/$NUCLEAR_FILE \
  --membrane-image $MOUNT_DIR/$MEMBRANE_FILE \
  --output-directory $MOUNT_DIR \
  --output-name mask.tif \
  --compartment whole-cell
```

## Using Docker

The script can also be run as a Docker image for improved portability.

### Build the image

```bash
docker build -t vanvalenlab/deepcell-applications .
```

### Run the image

For Docker API version >= 1.40:

```bash
export DATA_DIR=/path/to/data/dir
export MOUNT_DIR=/data
export NUCLEAR_FILE=example_nuclear_image.tif
export MEMBRANE_FILE=example_membrane_image.tif
docker run -it --gpus 1 \
-v $DATA_DIR:$MOUNT_DIR \
vanvalenlab/deepcell-applications:latest \
  --nuclear-image $MOUNT_DIR/$NUCLEAR_FILE \
  --membrane-image $MOUNT_DIR/$MEMBRANE_FILE \
  --output-directory $MOUNT_DIR \
  --output-name mask.tif \
  --compartment whole-cell
```

For Docker API version < 1.40:

```bash
export DATA_DIR=/path/to/data/dir
export MOUNT_DIR=/data
export NUCLEAR_FILE=example_nuclear_image.tif
export MEMBRANE_FILE=example_membrane_image.tif
NV_GPU=0 nvidia-docker run -it \
-v $DATA_DIR:$MOUNT_DIR \
vanvalenlab/deepcell-applications:latest \
  --nuclear-image $MOUNT_DIR/$NUCLEAR_FILE \
  --membrane-image $MOUNT_DIR/$MEMBRANE_FILE \
  --output-directory $MOUNT_DIR \
  --output-name mask.tif \
  --compartment whole-cell
```
