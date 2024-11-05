from ..payload.request.preprocess_text_request_dto import PreprocessTextRequestDto
from ..payload.response.preprocess_text_response_dto import PreprocessTextResponseDto
from ..util import preprocess_util


def preprocess_text_single(preprocess_text_request_dto: PreprocessTextRequestDto):
    preprocess_text_response_dto = preprocess_util.preprocess_text_single(preprocess_text_request_dto)
    return preprocess_text_response_dto
