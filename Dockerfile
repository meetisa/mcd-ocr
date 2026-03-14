FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
	tesseract-ocr \
	tesseract-ocr-ita \
	libgl1 \
	libglx-mesa0 \
	libglib2.0-0 \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src

VOLUME ["/data_input", "/data_output"]

ENV PYTHONPATH="/app"

CMD ["python", "-m", "src.main"]
