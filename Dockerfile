FROM python:3.10  

WORKDIR /app  

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./  
RUN pip install -r requirements.txt  

COPY . .  

EXPOSE 8501  

CMD ["streamlit", "run", "ocr_streamlit.py", "--server.port", "8501", "--server.address", "0.0.0.0"]