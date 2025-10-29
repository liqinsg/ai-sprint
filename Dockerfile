FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY day6_serve.py penguin_xgb_tuned.pkl mlruns.db ./
ENV MODEL_PATH=penguin_xgb_tuned.pkl
CMD ["uvicorn", "day6_serve:app", "--host", "0.0.0.0", "--port", "8000"]