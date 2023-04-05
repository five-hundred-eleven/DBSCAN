FROM python:3.11.2-slim-buster

# create non-root user
RUN useradd -m dev
USER dev

# use a virtualenv to avoid headaches with non-root users trying to use global
# pip etc
ENV BIN=/home/dev/venv/dbscan/bin
RUN pip install virtualenv
RUN python -m virtualenv /home/dev/venv/dbscan
# make sure we have the latest version of pip
RUN $BIN/pip install --upgrade pip

# copy our workspace
COPY . /home/dev/workspace/
WORKDIR /home/dev/workspace/

# install our deps
RUN $BIN/pip install -r requirements.txt

# start our app and listen
CMD $BIN/python -m gunicorn --bind :$PORT app:server
