FROM python:3.12-slim-bullseye

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["fastapi", "dev", "backend.py", "--host", "0.0.0.0", "--port", "5000"]
# CMD ["tail", "-f", "/dev/null"]