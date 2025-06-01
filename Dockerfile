FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords punkt_tab wordnet

COPY app/ app/
COPY models/ models/

EXPOSE 8501

CMD ["sh", "-c", "echo '🔗 App is running at: http://localhost:8501 Or http://127.0.0.1:8501' && streamlit run app/interactive_app.py --server.port=8501 --server.address=0.0.0.0"]
