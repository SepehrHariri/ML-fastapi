FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# add requirements file to image
COPY ./requirements.txt /app/requirements.txt

# install python libraries
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt


# add python code
COPY ./app /app/app/
