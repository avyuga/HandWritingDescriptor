FROM python:3.12-slim-bullseye

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
# CMD ["tail", "-f", "/dev/null"]