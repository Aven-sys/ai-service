import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fastapi import HTTPException
from typing import Optional
from ..payload.request.preprocess_text_request_dto import PreprocessTextRequestDto
from ..payload.response.preprocess_text_response_dto import PreprocessTextResponseDto

# Download NLTK stopwords (if not already available)
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define the preprocessing function
def preprocess_text_single(preprocess_text_request_dto: PreprocessTextRequestDto) -> PreprocessTextResponseDto:
    # Get the input text
    text = preprocess_text_request_dto.text
    preprocess_text = text  # Store the original text

    # Convert to lowercase if the flag is set
    if preprocess_text_request_dto.lowercase:
        text = text.lower()

    # Remove punctuation if the flag is set
    if preprocess_text_request_dto.remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove special characters if the flag is set
    if preprocess_text_request_dto.remove_special_characters:
        text = re.sub(r'[^A-Za-z0-9\s]+', '', text)

    # Remove stopwords if the flag is set
    if preprocess_text_request_dto.remove_stopwords:
        stop_words = set(stopwords.words("english"))
        text = " ".join(word for word in text.split() if word not in stop_words)

      # Apply lemmatization if the flag is set
    if preprocess_text_request_dto.lemmatize:
        text = " ".join(lemmatizer.lemmatize(word) for word in text.split())

    # Set the processed text as both preprocess_text and postprocess_text
    postprocess_text = text  # You can add further postprocessing steps here if needed

    # Create response DTO
    response_dto = PreprocessTextResponseDto(
        preprocess_text=preprocess_text,
        postprocess_text=postprocess_text
    )

    return response_dto
