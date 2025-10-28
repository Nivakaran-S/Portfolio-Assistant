# Use Python 3.10 with bookworm (newer sqlite)
FROM python:3.10-bookworm

WORKDIR /app

COPY requirements.txt /app/

# install deps
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /app

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
