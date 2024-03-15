#!/bin/bash

set -e

python3 -c "from API.myapp import app" # preheat, download model weights

gunicorn API.myapp:app -b $1