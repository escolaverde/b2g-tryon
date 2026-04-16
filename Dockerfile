FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
RUN mkdir -p uploads results models

EXPOSE 8000

CMD ["python", "-c", "import os; port = os.environ.get('PORT', '8000'); import uvicorn; uvicorn.run('app.main:app', host='0.0.0.0', port=int(port))"]
