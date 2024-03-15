#!/bin/bash

set -e

IMAGE="stamp-recognition-image"
NAME="stamp-recognition"
DATA_FILE=$(pwd)/data.db
OPTIONS="-d --name $NAME --mount type=bind,source=$DATA_FILE,target=/app/data.db  -p 8190:8190"

docker build . -t $IMAGE
docker stop $NAME || true && docker rm $NAME || true
docker run $OPTIONS $IMAGE
