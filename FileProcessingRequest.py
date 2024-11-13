from langserve import CustomUserType
from pydantic import Field

class FileProcessingRequest(CustomUserType):
    """Request including a base64 encoded file."""

    # The extra field is used to specify a widget for the playground UI.
    file: list = Field(...)
    file_name: list = Field(...)