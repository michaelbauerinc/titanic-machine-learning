FROM tensorflow/tensorflow


WORKDIR /workspace

COPY main.py requirements.txt .

RUN pip install -r requirements.txt

CMD ["python3", "main.py"]