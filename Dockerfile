FROM python:3.11
FROM node:19

#ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code

# Install python dependencies
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install node dependencies
COPY ./package*.json ./
RUN npm install

# Bundle 
COPY ./ ./

# Build tailwind
RUN npm tw-build

# Run app
CMD ["python", "./app.py"]