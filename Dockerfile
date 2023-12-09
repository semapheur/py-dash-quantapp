FROM python:3.11
FROM node:20

#ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code

# Install python dependencies
COPY ./requirements.txt ./
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install OCR
RUN supo apt install tesseract-ocr

# Install node dependencies
COPY ./package*.json ./
RUN npm install

# Bundle 
COPY ./ ./

# Build tailwind
RUN npm tw-build

EXPOSE 8080

# Run app
CMD ["python", "./app.py"]