FROM python:3.11-slim-buster

WORKDIR /app

COPY ./ /app

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y ffmpeg portaudio19-dev gcc libgomp1
# ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1



RUN pip install -r requirements.txt

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]