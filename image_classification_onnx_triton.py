import numpy as np
import onnxruntime as rt
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow import keras
import tritonclient.http as httpclient
from tritonclient.utils import *


def main():
    sess = rt.InferenceSession("./saved_model/model.onnx")
    data = keras.datasets.fashion_mnist
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    (train_images, train_labels), (test_images, test_labels) = data.load_data()
    # get_all_images(test_images)
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    target = np.array(test_images, dtype=np.float32)

    out = sess.run(["dense_1"], {"flatten_input": target.astype(np.float32)})
    prediction = out[0]

    client = httpclient.InferenceServerClient("localhost:8000")
    triton_inputs = [
        httpclient.InferInput("flatten_input", target.shape,
                              np_to_triton_dtype(target.dtype))
    ]
    triton_inputs[0].set_data_from_numpy(target)

    triton_outputs = [
        httpclient.InferRequestedOutput("dense_1")
    ]

    triton_response = client.infer("image_class_onnx", inputs=triton_inputs,
                                   request_id=str(1),
                                   outputs=triton_outputs)
    response = triton_response.as_numpy("dense_1")

    for i in range(5):
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel("Actual: " + class_names[test_labels[i]])
        plt.ylabel("Prediction: " + class_names[np.argmax(response[i])])
        plt.show()


def get_all_images(test_images):
    for i in range(test_images.shape[0]):
        im = Image.fromarray(np.uint8(test_images[i]))
        im.save('./images/{}.png'.format(i))
    plt.imsave('./images', test_images, cmap=plt.cm.binary)


if __name__ == '__main__':
    main()
