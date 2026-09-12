FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"

COPY artifacts/ ./artifacts/
COPY config/ ./config/
COPY src/ ./src/

ENV PYTHONPATH=/app
ENV PORT=3000

EXPOSE 3000

CMD ["python3", "-m", "src.app"]

