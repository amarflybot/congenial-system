import onnxruntime as rt
from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np


def main():
    sess = rt.InferenceSession("./saved_model/model.onnx")
    data = keras.datasets.fashion_mnist

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    (train_images, train_labels), (test_images, test_labels) = data.load_data()
    test_images = test_images / 255.0
    target = np.array(test_images, dtype=np.float32)

    out = sess.run(["dense_1"], {"flatten_input": target.astype(np.float32)})
    prediction = out[0]
    for i in range(5):
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel("Actual: " + class_names[test_labels[i]])
        plt.ylabel("Prediction: " + class_names[np.argmax(prediction[i])])
        plt.show()


if __name__ == '__main__':
    main()
