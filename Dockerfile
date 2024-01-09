# Get the python 3.8 base docker image
FROM python:3.8
RUN echo "\n Run Dockerfile"

COPY ./requirements.txt .

# Install pipenv
RUN python -m pip install -r requirements.txt
RUN python -m pip install flake8 pytest coveralls codecov pytest-cov