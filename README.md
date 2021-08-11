# deepcell-applications

A script and runnable Docker image for plugging DeepCell Applications (like `Mesmer`) into existing pipelines.

## Running the Python script

The `run_app.py` script is used to read the input files from the user and process them with the selected Application.
An example Python script `run_app.py` is provided as an example `deepcell.applications` workflow.

### Script arguments

The first required argument to the script is the Application name: `python run_app.py APP_NAME`.
Each supported application has a variety of different configuration arguments.
Below is a table summarizing the currently supported applications and their arguments and any defaults.
For more information, use `python run_app.py --help`.

#### Mesmer arguments

| Name | Description | Default Value |
| :--- | :--- | :--- |
| `--output-directory` | Directory to save output file. | `"./output"` |
| `--output-name` | The name for the output file. | `"mask.tif"` |
| `--nuclear-image` | **REQUIRED**: The path to a nuclear channel image. | `""` |
| `--membrane-image` | The path to a membrane channel image. | `""` |
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

This repository has published versions for both CPU and GPU for each versioned release.
However, it is easy to build a custom image to test any new functionality:

```bash
docker build -t vanvalenlab/deepcell-applications .
```

It is also possible to change the base DeepCell version when building the image, using the build-arg `DEEPCELL_VERSION`.
This makes it simple to build a CPU-only version of the image or to build a new version of `deepcell-tf`.

```bash
# the -gpu tag is required to enable GPU compatibility when overriding versions
docker build --build-arg DEEPCELL_VERSION=0.9.0-gpu -t vanvalenlab/deepcell-applications .
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
  vanvalenlab/deepcell-applications:latest-gpu \
  $APPLICATION \
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
export APPLICATION=mesmer
export NUCLEAR_FILE=DNA.tif
export MEMBRANE_FILE=Membrane.tif
docker run -it \
  -v $DATA_DIR:$MOUNT_DIR \
  vanvalenlab/deepcell-applications:latest-gpu \
  $APPLICATION \
  --nuclear-image $MOUNT_DIR/$NUCLEAR_FILE \
  --membrane-image $MOUNT_DIR/$MEMBRANE_FILE \
  --output-directory $MOUNT_DIR \
  --output-name mask.tif \
  --compartment whole-cell
```
