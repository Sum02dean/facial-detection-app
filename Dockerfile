From python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 7860

# Run the application
CMD [ "python", "app/app.py" ]
