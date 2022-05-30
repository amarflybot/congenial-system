from fastapi import Depends
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter

from app.service.hello_service import get_hello_service

router = InferringRouter()  # This is the router that decouples Controller class form the main class


@cbv(router)
class HelloController:

    def __init__(self, hello_service=Depends(get_hello_service)) -> None:
        self.hello_service = hello_service

    @router.get("/", tags=["hello"])
    def hello(self):
        return self.hello_service.get_hello()
