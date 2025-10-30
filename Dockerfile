FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip git libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/hpb-seg
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app

ENV APP_HOME=/opt/hpb-seg \
    PYTHONUNBUFFERED=1 \
    HPB_IN_ROOT=/tmp/hpb_in \
    HPB_OUT_ROOT=/tmp/hpb_out

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
