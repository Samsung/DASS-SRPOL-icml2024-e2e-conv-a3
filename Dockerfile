FROM nvcr.io/nvidia/tensorflow:23.04-tf2-py3
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
COPY requirements.txt .
RUN pip install -r requirements.txt