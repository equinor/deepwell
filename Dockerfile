FROM python:3.7

RUN apt-get update && apt-get install -y libopenmpi-dev

ENV PROJECT_PATH=/usr/src
WORKDIR $PROJECT_PATH

#Install requirements
COPY requirements.txt $PROJECT_PATH
RUN pip install --no-cache-dir -r $PROJECT_PATH/requirements.txt

#Install our custom openai gym
COPY env $PROJECT_PATH
RUN pip install --no-cache-dir -e $PROJECT_PATH/gym-dw


#Instead of running python directly, it is run after the container is started. This is to allow the tensorboard server to start first. See deepwellstart.ps1
#ENTRYPOINT [ "python","-u","/app/main.py" ]
#The "-u" is there to run it "unbuffered". That means that the printed output from python wil be shown in our host terminal, not the invisible container terminal



