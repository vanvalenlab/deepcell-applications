# Use vanvalenlab/deepcell-tf as the base image
# Change the build arg to edit the base image version.
ARG DEEPCELL_VERSION=0.5.0-gpu

FROM vanvalenlab/deepcell-tf:${DEEPCELL_VERSION}

WORKDIR /usr/src/app

COPY main.py .

ENTRYPOINT ["python", "main.py"]
