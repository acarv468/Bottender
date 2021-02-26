FROM python:3.8-slim

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 5001

ENTRYPOINT [ "python", "app.py" ]