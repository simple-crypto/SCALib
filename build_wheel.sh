#! /bin/bash

IMG=$(docker build -q - < Dockerfile) || exit
echo Docker image: $IMG
docker run --user $(id -u):$(id -g) --rm -v $(pwd):/io $IMG
