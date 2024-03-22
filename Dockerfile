# Get the python 3.10 base docker image
FROM python:3.10-slim as test-env

RUN echo "\n Run Dockerfile"

COPY  Makefile .
COPY ./requirements/*.txt  ./requirements

# Install pipenv
RUN apt-get -y update &&\
    apt-get -y install build-essential &&\
    make install &&\
    make install-test