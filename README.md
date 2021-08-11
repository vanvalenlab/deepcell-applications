# deepcell-applications

A runnable Docker image for plugging DeepCell Applications (like `Mesmer`) into existing pipelines.

## Running the Python script

An example Python script `run_app.py` is provided as an example `deepcell.applications` workflow.

### Script arguments

This script uses `argparse` to process command line arguments. Below is a table summarizing the arguments and any defaults. For more information, use `python run_app.py --help`.

| Name | Description | Default Value |
| :--- | :--- | :--- |
| `app` | **REQUIRED**: The name of the Application to use, must be one of: `["Mesmer"]` | `""` |
| `--nuclear-image` | **REQUIRED**: The path to a nuclear channel image. | `""` |
| `--membrane-image` | The path to a membrane channel image. | `""` |
| `--output-directory` | Directory to save output file. | `"./output"` |
| `--output-name` | The name for the output file. | `"mask.tif"` |
| `--compartment` | Predict nuclear or whole-cell segmentation. | `"whole-cell"` |

### Script command

```bash
export DATA_DIR=/path/to/data/dir
export MOUNT_DIR=/data
export APPLICATION=mesmer
export NUCLEAR_FILE=example_nuclear_image.tif
export MEMBRANE_FILE=example_membrane_image.tif
python run_app.py $APPLICATION \
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
export APPLICATION=mesmer
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
vanvalenlab/deepcell-applications:latest
  $APPLICATION \
  --nuclear-image $MOUNT_DIR/$NUCLEAR_FILE \
  --membrane-image $MOUNT_DIR/$MEMBRANE_FILE \
  --output-directory $MOUNT_DIR \
  --output-name mask.tif \
  --compartment whole-cell
```
