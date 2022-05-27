from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter

from app.service.hello_service import HelloService

router = InferringRouter()  # This is the router that decouples Controller class form the main class


@cbv(router)
class HelloController:

    def __init__(self, hello_service=HelloService()) -> None:
        self.hello_service = hello_service

    @router.get("/", tags=["root"])
    def hello(self):
        return self.hello_service.get_hello()
