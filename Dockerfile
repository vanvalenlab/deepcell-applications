# Use vanvalenlab/deepcell-tf as the base image
# Change the build arg to edit the base image version.
ARG DEEPCELL_VERSION=0.8.3-gpu

FROM vanvalenlab/deepcell-tf:${DEEPCELL_VERSION}

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Download and cache the model weights
RUN python -c "import deepcell; deepcell.applications.MultiplexSegmentation()"

ENTRYPOINT ["python", "main.py"]
