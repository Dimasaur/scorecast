FROM python:3.10.6-buster

COPY build-api /build-api
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .

# RUN CONTAINER LOCALLY
#CMD uvicorn build-api.api:app --host 0.0.0.0

# RUN CONTAINER DEPLOYED
CMD uvicorn build-api.api:app --host 0.0.0.0 --port $PORT
