FROM python:3.8

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Generate Model
COPY ./image_classification.py ./image_classification.py
RUN python3 image_classification.py
RUN python3 -m tf2onnx.convert --saved-model ./model --opset 15 --output /saved_model/model.onnx

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
