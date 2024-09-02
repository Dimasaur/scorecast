FROM python:3.10-bookworm

COPY build-api /build-api
COPY requirements.txt /requirements.txt


RUN pip install --upgrade pip
RUN pip install -r requirements.txt
