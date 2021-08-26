# Use vanvalenlab/deepcell-tf as the base image
# Change the build arg to edit the base image version.
ARG DEEPCELL_VERSION=0.10.0-gpu

FROM vanvalenlab/deepcell-tf:${DEEPCELL_VERSION}

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download and cache the model weights
RUN python -c "import deepcell_applications as dca; [v['class']() for v in dca.settings.VALID_APPLICATIONS.values()]"

ENTRYPOINT ["python", "run_app.py"]
