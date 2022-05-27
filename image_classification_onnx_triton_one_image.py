import numpy as np
import onnxruntime as rt
from matplotlib import pyplot as plt
from tensorflow import keras
import tritonclient.http as httpclient
from tritonclient.utils import *
from PIL import Image

def main():
    sess = rt.InferenceSession("./saved_model/model.onnx")
    data = keras.datasets.fashion_mnist
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    (train_images, train_labels), (test_images, test_labels) = data.load_data()
    # for i in range(test_images.shape[0]):
    #     im = Image.fromarray(np.uint8(test_images[i]))
    #     im.save('/home/amarendrakumar/PycharmProjects/learnML/images/{}.png'.format(i))

    # plt.imsave('/home/amarendrakumar/PycharmProjects/learnML/images', test_images, cmap=plt.cm.binary)
    test_image = np.asarray(Image.open("./images/9.png"))
    test_image = test_image / 255.0
    target = np.array([test_image], dtype=np.float32)

    out = sess.run(["dense_1"], {"flatten_input": target.astype(np.float32)})
    prediction = out[0]

    client = httpclient.InferenceServerClient("localhost:8000")
    print(target.shape)
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

    plt.grid(False)
    plt.imshow(test_image, cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[9]])
    plt.ylabel("Prediction: " + class_names[np.argmax(response[0])])
    plt.show()



if __name__ == '__main__':
    main()
