FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

WORKDIR /workspace

RUN apt update
RUN apt install -y make

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD [ "python", "test.py", "--dataroot", "/input", "--load_size", "512", "--output_dir", "/output", "--gpu_ids", "0"]
