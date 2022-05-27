# Deeplearning model to Triton

The motivation to crete this repo is to get a minimal deeplearning model running on Triton with onnxruntime backend.

## Install on local

```shell
$ python3 -m venv .
$ ./venv/bin.activate
$ pip install -r requirements.txt
```

## on docker-compose

```shell
## Installation
$ podman-compose run --rm build pip install -r requirements.txt


## Run tests
$ podman-compose run --rm build python -m unittest
```


## Run in steps

### 1. Simple tensorflow based Image Classification
This would create a tensorflow model in source dir.
Also, we would see minimal deeplearning tf model predict fashion minist.
File: [image_classification.py](image_classification.py)

### 2. Save the model as onnx
```shell
$ python -m tf2onnx.convert --saved-model ./model --opset 15 --output model.onnx
```

### 3. Load Onnx and use inference
Loads onnx model and uses onnxruntime to predict
File: [image_classification_onnx.py](image_classification_onnx.py)

### 4. Run triton with the model repository configuration
Triton Config: [triton_model_repository](./triton_model_repository)

