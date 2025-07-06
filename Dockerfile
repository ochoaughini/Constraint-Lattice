FROM python:3.11-slim

WORKDIR /

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV ENABLE_SAAS_FEATURES=false

CMD ["python", "examples/phi2_moderation_demo.py"]
