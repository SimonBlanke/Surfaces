# Get the python 3.10 base docker image
FROM python:3.10-slim
RUN echo "\n Run Dockerfile"

COPY requirements/requirements.in requirements/requirements-test.in Makefile ./

# Install pipenv
RUN make requirement &&\
    python -m pip install -r requirements.txt -r requirements-test.txt
