from fastapi_utils.api_model import APIModel


class ImageRequest(APIModel):
    data: str
    description: str
