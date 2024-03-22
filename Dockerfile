# Get the python 3.10 base docker image
FROM python:3.10-slim as test-env

COPY ./requirements/*.txt .

# Install pipenv
RUN RUN echo $(ls -1) &&\
    python -m pip install -r ./requirements/requirements.txt &&\
    python -m pip install -r ./requirements/requirements-test.txt
