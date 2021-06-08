#!/bin/bash

docker build --tag wordcount .
docker run --name wordcount -it wordcount
docker cp wordcount:/home ./