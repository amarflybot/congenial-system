import unittest
from unittest.mock import patch

from app.web.entities.image_request import ImageRequest
from app.web.predict_controller import ImageClassificationController


class TestImageClassificationController(unittest.TestCase):

    def test_should_return_formatted_shoe_if_service_returns(self):
        with patch('app.service.image_classification_model_service.ImageClassificationService') as mock:
            instance = mock.return_value
            instance.predict.return_value = "shirt"
            _predict_controller = ImageClassificationController(instance)
            response = _predict_controller.predict(image_request=ImageRequest(data="testData", description="test"))
            self.assertEqual({'name': 'shirt'}, response)
