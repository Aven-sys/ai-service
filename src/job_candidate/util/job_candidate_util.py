import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fastapi import HTTPException
from typing import Optional, List
from sentence_transformers import SentenceTransformer, util
from ..payload.response.general_matching_response import GeneralMatchingResponse, MatchResult
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK stopwords (if not already available)
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# def preprocess_text_words(text: str) -> str:
#     # Lowercase, remove punctuation and stopwords, and lemmatize
#     text = text.lower()
#     text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
#     words = text.split()
#     words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words("english")]
#     return " ".join(words)

def preprocess_text_words(text: str) -> str:
    # Lowercase, remove punctuation and stopwords, and lemmatize
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    words = nltk.word_tokenize(text)
    return words

def preprocess_text_sentences(text: str) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    return sentences
    # return [(sentence) for sentence in sentences]

def get_skills_matching(job_skills: List[str], candidate_skills: List[str], threshold: float = 0.5, model_name: Optional[str] = "all-MiniLM-L12-v2") -> GeneralMatchingResponse:
    # Preprocess job and candidate skills
    all_skills = [(skill) for skill in (job_skills + candidate_skills)]
    
    # Load model (or reuse if pre-loaded)
    model = SentenceTransformer(model_name)

    # Encode all skills in a single batch
    all_embeddings = model.encode(all_skills, convert_to_tensor=True)
    
    # Split embeddings for job and candidate skills
    job_embeddings = all_embeddings[:len(job_skills)]
    candidate_embeddings = all_embeddings[len(job_skills):]
    
    # Calculate cosine similarities between all pairs
    cosine_similarities = util.pytorch_cos_sim(job_embeddings, candidate_embeddings)
    
    # Collect matching results above threshold
    results = []
    total_matches = 0

    for i, job_skill in enumerate(job_skills):
        for j, candidate_skill in enumerate(candidate_skills):
            similarity = cosine_similarities[i][j].item()
            if similarity >= threshold:
                results.append(MatchResult(
                    source_item=job_skill,
                    target_item=candidate_skill,
                    similarity_score=similarity
                ))
                total_matches += 1

    # Calculate the overall score out of 10 based on matches for job skills
    overall_score = min(10, (total_matches / len(job_skills)) * 10) if job_skills else 0

    return GeneralMatchingResponse(matches=results, overall_score=overall_score)

def get_tfidf_skills_matching(job_skills: List[str], candidate_skills: List[str], threshold: float = 0.5) -> GeneralMatchingResponse:
    # Vectorize the job skills
    vectorizer = TfidfVectorizer()
    job_embeddings = vectorizer.fit_transform(job_skills)
    
    # Transform the candidate skills using the same vectorizer (trained on job skills)
    candidate_embeddings = vectorizer.transform(candidate_skills)
    
    # Calculate cosine similarities between job and candidate skills
    cosine_similarities = cosine_similarity(job_embeddings, candidate_embeddings)
    
    # Collect matching results above threshold
    results = []
    total_matches = 0

    for i, job_skill in enumerate(job_skills):
        for j, candidate_skill in enumerate(candidate_skills):
            similarity = cosine_similarities[i][j]
            if similarity >= threshold:
                results.append(MatchResult(
                    source_item=job_skill,
                    target_item=candidate_skill,
                    similarity_score=similarity
                ))
                total_matches += 1

    # Calculate the overall score out of 10 based solely on matches for job skills
    overall_score = (total_matches / len(job_skills)) * 10 if job_skills else 0

    # Return in the required GeneralMatchingResponse format
    return GeneralMatchingResponse(matches=results, overall_score=min(overall_score, 10))


def get_tfidf_skills_matching2(job_skills: List[str], candidate_skills: List[str], threshold: float = 0.5) -> GeneralMatchingResponse:
    # Vectorize the job skills
    vectorizer = TfidfVectorizer()
    job_embeddings = vectorizer.fit_transform(job_skills)
    
    # Transform the candidate skills using the same vectorizer (trained on job skills)
    candidate_embeddings = vectorizer.transform(candidate_skills)
    
    # Calculate cosine similarities between job and candidate skills
    cosine_similarities = cosine_similarity(job_embeddings, candidate_embeddings)
    
    # Collect matching results above threshold and track unique job skill matches
    results = []
    matched_job_indices = set()  # To track matched job qualifications

    for i, job_skill in enumerate(job_skills):
        for j, candidate_skill in enumerate(candidate_skills):
            similarity = cosine_similarities[i][j]
            if similarity >= threshold and i not in matched_job_indices:
                results.append(MatchResult(
                    source_item=job_skill,
                    target_item=candidate_skill,
                    similarity_score=similarity
                ))
                matched_job_indices.add(i)  # Mark this job skill as matched

    # Calculate the overall score out of 10 based solely on unique matches for job skills
    overall_score = (len(matched_job_indices) / len(job_skills)) * 10 if job_skills else 0

    # Return in the required GeneralMatchingResponse format
    return GeneralMatchingResponse(matches=results, overall_score=min(overall_score, 10))

def get_years_of_experience_matching(job_years_of_experience, candidate_years_of_experience):
    if (job_years_of_experience == 0 or job_years_of_experience == None):
        return GeneralMatchingResponse.empty_response()
    
    if (candidate_years_of_experience == None): 
        return GeneralMatchingResponse(matches=None, overall_score=0)
    
    if (candidate_years_of_experience >= job_years_of_experience):
        return GeneralMatchingResponse(matches=None, overall_score=10)
    elif (candidate_years_of_experience < job_years_of_experience):
        score = (candidate_years_of_experience / job_years_of_experience) * 10
        return GeneralMatchingResponse(matches=None, overall_score=score)
    

       