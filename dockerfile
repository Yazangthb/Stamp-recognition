FROM python:3.10-slim-buster
RUN apt update --no-install-recommends -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
EXPOSE 8190
RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
COPY . /app
CMD ["./run.sh", "0.0.0.0:8190"]
