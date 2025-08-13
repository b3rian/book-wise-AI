FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./streamlit_app/

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    curl \
 && rm -rf /var/lib/apt/lists/*

EXPOSE $PORT

CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
