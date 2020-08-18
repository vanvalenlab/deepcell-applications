# deepcell-applications

A template of how to create and combine deep learning workflows using `deepcell-tf` and Docker.

## Build the image

```bash
docker build -t vanvalenlab/deepcell-applications .
```

## Run the image

```bash
docker run --gpus=1 -it \
-v /path/to/data/folder:/data \
vanvalenlab/deepcell-applications:latest /data/$FILEPATH
```
