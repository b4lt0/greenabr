FROM python:3.11.5-bookworm

WORKDIR /root

COPY . .

RUN pip install -r requirements.txt