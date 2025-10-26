FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY day6_serve.py .
COPY penguin_auto.pkl .
ENV MODEL_PATH=penguin_auto.pkl
CMD ["uvicorn", "day6_serve:app", "--host", "0.0.0.0", "--port", "8000"]
