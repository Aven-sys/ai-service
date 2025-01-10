import re
import string
import torch
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fastapi import HTTPException
from typing import Optional, List
from sentence_transformers import SentenceTransformer, util, SimilarityFunction
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
    

# --------------------------------Zave's Code --------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
semantic_model_name = 'paraphrase-MiniLM-L3-v2'
semantic_model = SentenceTransformer(semantic_model_name).to(device)

def calculate_semantic_scores_batch(base_data: List[str], compare_data: List[str], threshold: float) -> torch.Tensor:
    if not base_data or not compare_data:
        return GeneralMatchingResponse(matches=[], overall_score=0)

    # Preprocess base and compare data
    base_data = [item.lower() for item in base_data]
    compare_data = [item.lower() for item in compare_data]
    all_data = base_data + compare_data

    # Encode all data in a single batch
    try:
        all_embeddings = semantic_model.encode(all_data, convert_to_tensor=True)
    except Exception as e:
        print(f"Error encoding data with model '{model_name}': {e}")
        return GeneralMatchingResponse(matches=[], overall_score=0)

    # Split embeddings for base and compare data
    base_embeddings = all_embeddings[:len(base_data)]
    compare_embeddings = all_embeddings[len(base_data):]
    
    # Calculate cosine similarities between all pairs
    try:
        cosine_similarities = util.pytorch_cos_sim(base_embeddings, compare_embeddings)
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return GeneralMatchingResponse(matches=[], overall_score=0)
    
    # Collect matching results above threshold
    results = []
    total_matches = 0

    for i, base_item in enumerate(base_data):
        for j, compare_item in enumerate(compare_data):
            similarity = cosine_similarities[i][j].item()
            if similarity >= threshold:
                results.append(MatchResult(
                    source_item=base_item,
                    target_item=compare_item,
                    similarity_score=similarity
                ))
                # print(f"Match: base Skill '{base_data}' - compare Skill '{compare_data}' with Similarity Score: {similarity:.3f}")
                total_matches += 1

    # Calculate the overall score out of 10 based on matches for base skills
    overall_score = min(1, (total_matches / len(base_data)) * 1) if base_data else 0
    
    return GeneralMatchingResponse(matches=results, overall_score=overall_score)

def get_skills_matching_v2(base_skills: List[str], compare_skills: List[str]) -> GeneralMatchingResponse: 
    semantic_response = calculate_semantic_scores_batch(base_skills, compare_skills, 0.5)
    return semantic_response

def get_job_title_matching(base_job_title: List[str], compare_job_title: List[str]) -> GeneralMatchingResponse:
    semantic_response = calculate_semantic_scores_batch(base_job_title, compare_job_title, 0.5)
    return semantic_response

def get_location_matching(base_location: List[str], compare_location: List[str]) -> GeneralMatchingResponse:
    semantic_response = calculate_semantic_scores_batch(base_location, compare_location, 0.65)
    return semantic_response

def get_keyword_matching(base_keywords: List[str], compare_keywords: List[str]) -> GeneralMatchingResponse:
    semantic_response = calculate_semantic_scores_batch(base_keywords, compare_keywords, 0.5)
    return semantic_response

def get_salary_matching(base_salary: List[str], compare_salary: List[str]) -> GeneralMatchingResponse:
    # Extract numeric value from base_salary string
    base_salary_match = re.search(r"[\d,]+", base_salary[0]) if base_salary else 0
    base_salary_float = float(base_salary_match.group().replace(",", "")) if base_salary_match else 0

    # Extract numeric value from compare_salary string
    compare_salary_match = re.search(r"[\d,]+", compare_salary[0]) if compare_salary else 0
    compare_salary_float = float(compare_salary_match.group().replace(",", "")) if compare_salary_match else 0

    # Initialize matching results list
    matching_results = []

    # Check if base salary is greater than or equal to compare salary
    if base_salary_float >= compare_salary_float:
        similarity_score = 1.0
    else:
        similarity_score = base_salary_float / compare_salary_float if base_salary_float != 0 else 0

    # Append match result
    matching_results.append(
        MatchResult(
            source_item=str(base_salary_float),
            target_item=str(compare_salary_float),
            similarity_score=similarity_score
        )
    )

    # Return the matching response
    return GeneralMatchingResponse(matches=matching_results, overall_score=similarity_score)

def get_years_experience_matching(base_years_of_experience: List[str], compare_years_of_experience: List[str]) -> GeneralMatchingResponse:
    # Extract numeric value from string
    base_years_of_experience = int(base_years_of_experience[0]) if base_years_of_experience else None
    compare_years_of_experience = int(compare_years_of_experience[0]) if compare_years_of_experience else None

    # Check if base years of experience is zero or None
    if base_years_of_experience is None or compare_years_of_experience is None:
        return GeneralMatchingResponse(matches=None, overall_score=0)

    # Calculate the overall score based on the ratio of compare years to base years
    if compare_years_of_experience >= base_years_of_experience:
        overall_score = 1
    else:
        overall_score = (compare_years_of_experience / base_years_of_experience) * 1

    matches = []
    matches.append(MatchResult(
        source_item=str(base_years_of_experience), 
        target_item=str(compare_years_of_experience), 
        similarity_score=overall_score
    ))

    # Return the matching response
    return GeneralMatchingResponse(matches=matches, overall_score=overall_score)

def get_experience_matching(base_experience: List[str], compare_experience: List[str]) -> GeneralMatchingResponse:
    semantic_response = calculate_semantic_scores_batch(base_experience, compare_experience, 0.5)
    return semantic_response

def get_qualification_matching(base_qualifications: List[str], compare_qualifications: List[str]) -> GeneralMatchingResponse:
    semantic_response = calculate_semantic_scores_batch(base_qualifications, compare_qualifications, 0.5)
    return semantic_response