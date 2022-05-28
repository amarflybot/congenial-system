import base64
import unittest

from app.service.image_classification_model_service import ImageClassificationService


class TestImageClassificationService(unittest.TestCase):

    def test_should_predict_as_sneaker(self):
        with open("./test/service/9.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            image_classification_model_service = ImageClassificationService()
            image_type = image_classification_model_service.predict(data=encoded_string)
            self.assertEqual("Sneaker", image_type)