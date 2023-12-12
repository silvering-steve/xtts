FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

#  No interaction in install
ARG DEBIAN_FRONTEND=noninteractive

# Install base utils
RUN apt-get update \
    && apt-get install -y procps libsm6 libxext6 libxrender-dev libglib2.0-0 \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Change working directory
WORKDIR /app

# Install requirment
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]