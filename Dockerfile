FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
#COPY day6_serve.py .
ENV MLFLOW_TRACKING_URI=https://your-mlflow-server.com  # or keep local file
CMD ["uvicorn", "day6_serve:app", "--host", "0.0.0.0", "--port", "8000"]
