FROM python:3.7

RUN apt-get update && apt-get install -y \
  libopenmpi-dev \
  telnet

ENV PROJECT_PATH=/usr/src
WORKDIR $PROJECT_PATH

#Install requirements
COPY requirements.txt $PROJECT_PATH
RUN pip install --no-cache-dir -r $PROJECT_PATH/requirements.txt

EXPOSE 8050

ENTRYPOINT [ "gunicorn","-w","4","-b","0.0.0.0:8050","app:server","--chdir","/usr/src/app" ]
#ENTRYPOINT [ "python" ]



