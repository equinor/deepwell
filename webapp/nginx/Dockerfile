FROM python:3.7

RUN apt-get update && apt-get install -y \
  nginx \
  telnet

ENV PROJECT_PATH=/usr/src
WORKDIR $PROJECT_PATH

#Install requirements
#COPY requirements.txt $PROJECT_PATH
#RUN pip install --no-cache-dir -r $PROJECT_PATH/requirements.txt

RUN rm /etc/nginx/sites-enabled/default
COPY deepwell_nginx /etc/nginx/sites-enabled/

CMD ["nginx","-g","daemon off;"]




