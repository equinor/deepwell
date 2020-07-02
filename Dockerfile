    FROM python:3.7

    WORKDIR /usr/src/app
    
    #Install requirements
    COPY requirements.txt ./
    RUN pip install --no-cache-dir -r requirements.txt


    #Install our custom openai gym
    COPY env ./
    RUN pip install --no-cache-dir -e ./gym-ww

    CMD [ "python","-u","/app/main.py" ]

    #The "-u" is there to run it "unbuffered". That means that the printed output from python wil be shown in our host terminal, not the invisible container terminal