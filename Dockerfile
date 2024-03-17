# Get the python 3.10 base docker image
FROM python:3.10-slim
RUN echo "\n Run Dockerfile"

COPY requirements/ Makefile ./

# Install pipenv
RUN apt-get -y update &&\
    apt-get -y install build-essential &&\
    python -m pip install pip-tools &&\
    make requirement &&\
    make install &&\
    make install-test