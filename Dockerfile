FROM python:3.10-bookworm

COPY build-api /build-api
COPY requirements.txt /requirements.txt


RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn build-api.api:app --host 0.0.0.0 --port $PORT
