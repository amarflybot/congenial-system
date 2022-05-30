from fastapi import Request


class HelloService:
    def get_hello(self):
        return {"hello": "world"}


# Helper to grab dependencies that live in the app.state
def get_hello_service(request: Request) -> HelloService:
    # See application for the key name in `app`.
    return request.app.state.hello_service
