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

# tfidf_vectorizer = TfidfVectorizer(max_df=0.8, sublinear_tf=True)
tfidf_vectorizer = TfidfVectorizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
semantic_model_name = 'all-MiniLM-L12-v2'
semantic_model = SentenceTransformer(semantic_model_name).to(device)
semantic_model.similarity_fn_name = SimilarityFunction.DOT_PRODUCT
tokenizer = semantic_model.tokenizer
max_seq_length = semantic_model.max_seq_length
cls_token_id = torch.tensor([tokenizer.cls_token_id])
sep_token_id = torch.tensor([tokenizer.sep_token_id])

def calculate_tfidf_scores(job_data: List[str], candidate_data: List[str]) -> torch.Tensor:
    if not job_data or not candidate_data:
        return GeneralMatchingResponse(matches=[], overall_score=0)
    
    # Preprocess job and candidate data
    job_data = [item.lower() for item in job_data]
    candidate_data = [item.lower() for item in candidate_data]

    # Combine job and candidate datas
    all_data = job_data + candidate_data

    # Calculate TF-IDF scores
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_data)

    # For console output
    cosine_similarity_output = cosine_similarity(tfidf_matrix[:len(job_data)], tfidf_matrix[len(job_data):])
    match_threshold = 0.5
    matched_pairs = []
    for job_idx, job_score in enumerate(cosine_similarity_output):
        for candidate_idx, score in enumerate(job_score):
            if score >= match_threshold:
                job_item = job_data[job_idx]
                candidate_item = candidate_data[candidate_idx]
                matched_pairs.append(MatchResult(
                    source_item=job_item,
                    target_item=candidate_item,
                    similarity_score=score
                ))
                # print(f"Match: Job Item '{job_item}' - Candidate Item '{candidate_item}' with Similarity Score: {score:.3f}")

    # Calculate the overall score out of 10 based on matches for job data
    overall_score = min(1, (len(matched_pairs) / len(job_data)) * 1) if job_data else 0
    return GeneralMatchingResponse(matches=matched_pairs, overall_score=overall_score)

def calculate_keyword_scores(job_data: List[str], candidate_data: List[str]) -> torch.Tensor:
    if not job_data or not candidate_data:
        return GeneralMatchingResponse(matches=[], overall_score=0)
    
    # Preprocess job and candidate data
    job_data = [item.lower() for item in job_data]
    candidate_data = [item.lower() for item in candidate_data]

    matching_keywords_list = []
    for candidate_item in candidate_data:
        for keyword in job_data:
            if keyword in candidate_item:
                matching_keywords_list.append(MatchResult(
                    source_item=keyword,
                    target_item=keyword,
                    similarity_score=1
                ))

    keyword_score = len(matching_keywords_list) / len(job_data)

    # print(f"Keyword Matches: {matching_keywords_list}")
    # print(f"Keyword Score: {keyword_score}")

    return GeneralMatchingResponse(matches=matching_keywords_list, overall_score=keyword_score)

def calculate_semantic_scores_chunk(job_data: List[str], candidate_data: List[str], model_name: str = semantic_model_name) -> torch.Tensor:
    if not job_data or not candidate_data:
        return GeneralMatchingResponse(matches=[], overall_score=0)
    
    if model_name:
        semantic_model = SentenceTransformer(model_name).to(device)

    semantic_model.similarity_fn_name = SimilarityFunction.DOT_PRODUCT
    tokenizer = semantic_model.tokenizer
    max_seq_length = semantic_model.max_seq_length
    cls_token_id = torch.tensor([tokenizer.cls_token_id])
    sep_token_id = torch.tensor([tokenizer.sep_token_id])

    # Preprocess job and candidate data
    job_data = [item.lower() for item in job_data]
    candidate_data = [item.lower() for item in candidate_data]
         
    def tokenize_and_chunk(text):
        """Tokenize text, truncate to embedding_seq_length, and chunk within max sequence length."""
        tokens = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_seq_length,
            padding=True
        )
        input_ids, attention_mask = tokens['input_ids'][0], tokens['attention_mask'][0]

        # Define chunk size, accounting for [CLS] and [SEP] tokens
        chunk_size = max_seq_length - 2
        input_id_chunks = [input_ids[i:i + chunk_size] for i in range(0, len(input_ids), chunk_size)]
        attention_mask_chunks = [attention_mask[i:i + chunk_size] for i in range(0, len(attention_mask), chunk_size)]

        # Add [CLS] and [SEP] tokens to each chunk
        input_id_chunks = [torch.cat([cls_token_id, chunk, sep_token_id]) for chunk in input_id_chunks]
        attention_mask_chunks = [torch.cat([torch.tensor([1]), chunk, torch.tensor([1])]) for chunk in attention_mask_chunks]

        return input_id_chunks, attention_mask_chunks

    def get_normalized_average_embedding(text):
        """Calculate normalized mean embedding from text chunks."""
        input_id_chunks, attention_mask_chunks = tokenize_and_chunk(text)
        if not input_id_chunks:
            return np.zeros(semantic_model.get_sentence_embedding_dimension())

        # Accumulate embeddings to calculate the mean
        embedding_sum = 0
        for input_ids, attention_mask in zip(input_id_chunks, attention_mask_chunks):
            inputs = {'input_ids': input_ids.unsqueeze(0), 'attention_mask': attention_mask.unsqueeze(0)}
            with torch.no_grad():
                model_output = semantic_model(inputs, output_hidden_states=False, output_attentions=False)
                chunk_embedding = model_output['sentence_embedding']
            embedding_sum += chunk_embedding.cpu().numpy().squeeze()

        # Calculate mean embedding and normalize
        mean_embedding = embedding_sum / len(input_id_chunks)
        norm = np.linalg.norm(mean_embedding)
        return mean_embedding / norm if norm != 0 else mean_embedding

    # Obtain normalized embeddings for CV and job descriptions
    job_embeddings = [get_normalized_average_embedding(job_skill) for job_skill in job_data]
    candidate_embeddings = [get_normalized_average_embedding(candidate_skill) for candidate_skill in candidate_data]

    # Compute cosine similarity scores as dot products
    similarity_scores = []
    for job_emb in job_embeddings:
        for candidate_emb in candidate_embeddings:
            similarity_scores.append(np.dot(job_emb, candidate_emb))

    # Print matches with similarity scores above threshold
    matched_pairs = []
    for job_idx, job_skill in enumerate(job_data):
        for candidate_idx, candidate_skill in enumerate(candidate_data):
            score = similarity_scores[job_idx * len(candidate_data) + candidate_idx]
            if score >= 0.5:
                matched_pairs.append(MatchResult(
                    source_item=job_skill,
                    target_item=candidate_skill,
                    similarity_score=score
                ))
                print(f"Match: Job Skill '{job_skill}' - Candidate Skill '{candidate_skill}' with Similarity Score: {score:.3f}")

    # Calculate the overall score out of 10 based on matches for job data
    overall_score = min(1, (len(matched_pairs) / len(job_data)) * 1) if job_data else 0

    return GeneralMatchingResponse(matches=matched_pairs, overall_score=overall_score)

def calculate_semantic_scores_batch(job_data: List[str], candidate_data: List[str], model_name: str = semantic_model_name) -> torch.Tensor:
    if not job_data or not candidate_data:
        return GeneralMatchingResponse(matches=[], overall_score=0)
    
    if model_name:
        semantic_model = SentenceTransformer(model_name).to(device)

    # Preprocess job and candidate data
    job_data = [item.lower() for item in job_data]
    candidate_data = [item.lower() for item in candidate_data]
    all_data = job_data + candidate_data

    # Encode all data in a single batch
    all_embeddings = semantic_model.encode(all_data, convert_to_tensor=True)

    # Split embeddings for job and candidate data
    job_embeddings = all_embeddings[:len(job_data)]
    candidate_embeddings = all_embeddings[len(job_data):]
    
    # Calculate cosine similarities between all pairs
    cosine_similarities = util.pytorch_cos_sim(job_embeddings, candidate_embeddings)
    
    # Collect matching results above threshold
    results = []
    total_matches = 0

    for i, job_item in enumerate(job_data):
        for j, candidate_item in enumerate(candidate_data):
            similarity = cosine_similarities[i][j].item()
            if similarity >= 0.5:
                results.append(MatchResult(
                    source_item=job_item,
                    target_item=candidate_item,
                    similarity_score=similarity
                ))
                # print(f"Match: Job Skill '{job_data}' - Candidate Skill '{candidate_data}' with Similarity Score: {similarity:.3f}")
                total_matches += 1

    # Calculate the overall score out of 10 based on matches for job skills
    overall_score = min(1, (total_matches / len(job_data)) * 1) if job_data else 0
    
    return GeneralMatchingResponse(matches=results, overall_score=overall_score)

def get_skills_matching_v2(job_skills: List[str], candidate_skills: List[str]) -> GeneralMatchingResponse: 
    semantic_response = calculate_semantic_scores_batch(job_skills, candidate_skills, "paraphrase-MiniLM-L3-v2")
    return semantic_response

def get_job_title_matching(job_title: List[str], candidate_job_title: List[str]) -> GeneralMatchingResponse:
    semantic_response = calculate_semantic_scores_batch(job_title, candidate_job_title, "paraphrase-MiniLM-L3-v2")
    return semantic_response

def get_location_matching(job_location: List[str], candidate_location: List[str]) -> GeneralMatchingResponse:
    semantic_response = calculate_semantic_scores_batch(job_location, candidate_location, "paraphrase-MiniLM-L3-v2")
    return semantic_response

def get_keyword_matching(job_keywords: List[str], candidate_keywords: List[str]) -> GeneralMatchingResponse:
    semantic_response = calculate_semantic_scores_batch(job_keywords, candidate_keywords, "paraphrase-MiniLM-L3-v2")
    return semantic_response

def get_salary_matching(job_salary: str, candidate_salary: str) -> GeneralMatchingResponse:
    # Extract numeric value from job_salary string
    job_salary_match = re.search(r"[\d,]+", job_salary)
    job_salary_float = float(job_salary_match.group().replace(",", "")) if job_salary_match else 0

    # Extract numeric value from candidate_salary string
    candidate_salary_match = re.search(r"\d+(\.\d+)?", candidate_salary)
    candidate_salary_float = float(candidate_salary_match.group()) if candidate_salary_match else 0

    # Initialize matching results list
    matching_results = []

    # Check if job salary is greater than or equal to candidate salary
    if job_salary_float >= candidate_salary_float:
        similarity_score = 1.0
    else:
        similarity_score = job_salary_float / candidate_salary_float if job_salary_float != 0 else 0

    # Append match result
    matching_results.append(
        MatchResult(
            source_item=str(job_salary_float),
            target_item=str(candidate_salary_float),
            similarity_score=similarity_score
        )
    )

    # Return the matching response
    return GeneralMatchingResponse(matches=matching_results, overall_score=similarity_score)

def get_years_experience_matching(job_years_of_experience: int, candidate_years_of_experience: int) -> GeneralMatchingResponse:
    # Check if job years of experience is zero or None
    if job_years_of_experience is None or candidate_years_of_experience is None:
        return GeneralMatchingResponse(matches=None, overall_score=0)

    # Calculate the overall score based on the ratio of candidate years to job years
    if candidate_years_of_experience >= job_years_of_experience:
        overall_score = 1
    else:
        overall_score = (candidate_years_of_experience / job_years_of_experience) * 1

    matches = []
    matches.append(MatchResult(
        source_item=str(job_years_of_experience), 
        target_item=str(candidate_years_of_experience), 
        similarity_score=overall_score
    ))

    # Return the matching response
    return GeneralMatchingResponse(matches=matches, overall_score=overall_score)

def get_experience_matching(job_experience: List[str], candidate_experience: List[str]) -> GeneralMatchingResponse:
    semantic_response = calculate_semantic_scores_batch(job_experience, candidate_experience, "paraphrase-MiniLM-L3-v2")
    return semantic_response

def get_qualification_matching(job_qualifications: List[str], candidate_qualifications: List[str]) -> GeneralMatchingResponse:
    semantic_response = calculate_semantic_scores_batch(job_qualifications, candidate_qualifications, "paraphrase-MiniLM-L3-v2")
    return semantic_response