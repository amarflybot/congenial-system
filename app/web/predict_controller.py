from fastapi import Depends
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter

from app.service.image_classification_model_service import ImageClassificationService
from app.web.entities.image_request import ImageRequest

router = InferringRouter()  # Step 1: Create a router


@cbv(router)
class ImageClassificationController:

    def __init__(self, image_classification_service=Depends(ImageClassificationService)) -> None:
        self.image_classification_service = image_classification_service

    @router.post("/v1/imageClassifier", tags=["image_classification"])
    def predict(self, image_request: ImageRequest):
        predict = self.image_classification_service.predict(image_request.data)
        image_response = {"name": predict}
        return image_response
