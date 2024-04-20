FROM python:3.11

EXPOSE 8080
ENV PORT 8080

RUN groupadd -g 1000 userweb && \
    useradd -r -u 1000 -g userweb userweb

WORKDIR /home
RUN chown userweb:userweb /home

RUN apt-get update
RUN apt-get install poppler-utils tesseract-ocr -y
# needed by python module cv2
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

USER userweb

COPY . /home
RUN pip install -r /home/requirements.txt

RUN mkdir -p /home/index /home/db

CMD python /home/predictions.py