import os

import numpy as np
from PIL import Image
from io import BytesIO
import base64

import onnxruntime as rt


class ImageClassificationService:

    def __init__(self) -> None:
        self.sess = rt.InferenceSession("/saved_model/model.onnx", providers=['CPUExecutionProvider'])
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def predict(self, data: str):
        im = Image.open(BytesIO(base64.b64decode(data)))
        test_image = np.asarray(im)
        test_image = test_image / 255.0
        target = np.array([test_image], dtype=np.float32)
        out = self.sess.run(["dense_1"], {"flatten_input": target.astype(np.float32)})
        prediction = out[0]
        return self.class_names[np.argmax(prediction)]
