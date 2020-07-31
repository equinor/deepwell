# DeepWell Webapp
 
This is a stand-alone simple web application that lets you plot points in the environment, and generate a wellpath based on a trained model.


## How to run server

You can simply try the development server by installing the requirements and running app.py

To run the fully deployed server with gunicorn and nginx, you need to build the docker-compose file with Docker using

    docker-compose build

And then run it with

    docker-compose up

To change the ip that the server is running on, you have to change server_name in /nginx/deepwell_nginx