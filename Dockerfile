FROM python:3.7

RUN apt-get update && apt-get install -y libopenmpi-dev 



WORKDIR /usr/src/app

#Install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

#To be able to load model, should prob remove later but weren't able to figure out different solutio
COPY ppo2_shortpath.zip ./


#Install our custom openai gym
COPY env ./
RUN pip install --no-cache-dir -e ./gym-dw

ENTRYPOINT [ "python","-u","/app/main.py" ]

#The "-u" is there to run it "unbuffered". That means that the printed output from python wil be shown in our host terminal, not the invisible container terminal