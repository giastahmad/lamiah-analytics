FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod -R 777 /app

CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "main:app"]