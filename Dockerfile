FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# MediaPipe needs OpenGL ES/EGL runtime libraries even when the analyzer uses
# the CPU delegate. Without libgles2, image analysis fails with:
# libGLESv2.so.2: cannot open shared object file: No such file or directory
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libegl1 \
        libgles2 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
