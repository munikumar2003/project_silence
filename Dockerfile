FROM python:3.10-slim

WORKDIR /

COPY . .

#RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=5050

CMD ["gunicorn", "-b", "0.0.0.0:5050", "app"]