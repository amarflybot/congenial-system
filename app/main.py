from fastapi import FastAPI

from app.service.image_classification_model_service import ImageClassificationService
from app.service.hello_service import HelloService
from app.web.hello_controller import router as hello_router
from app.web.predict_controller import router as predict_router

app = FastAPI()

app.include_router(hello_router)
app.include_router(predict_router)


@app.on_event('startup')
def load_single_instance_service():
    app.state.image_classification_service = ImageClassificationService()
    app.state.hello_service = HelloService()
