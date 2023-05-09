FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

USER root

WORKDIR /app

RUN apt-get update && \
	apt install build-essential --yes && \
    apt-get install ffmpeg --yes && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install -r requirements.txt
RUN pip cache purge

EXPOSE 7860

CMD python run.py