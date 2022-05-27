from fastapi import FastAPI

from app.web.hello_controller import router as hello_router
from app.web.predict_controller import router as predict_router

app = FastAPI()

app.include_router(hello_router)
app.include_router(predict_router)
