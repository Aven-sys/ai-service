from ..payload.request.job_to_candidates_matching_request_dto import JobToCandidatesMatchingRequestDTO, JobParsedDataDTO, CandidateParsedDataDTO
from ..util import job_candidate_util
from common.util.langchain_pydantic_model_generator import print_pydantic_instance
from concurrent.futures import ThreadPoolExecutor, as_completed
import nltk
from typing import List

def get_job_to_candidates_matching(job_to_candidates_matching_request_Dto: JobToCandidatesMatchingRequestDTO):
    job_data = job_to_candidates_matching_request_Dto.jobData
    candidates_data = job_to_candidates_matching_request_Dto.candidatesData
    
    # To store results from each candidate matching
    results = []
    
    # Define the maximum number of workers (threads)
    max_workers = min(10, len(candidates_data))  # Adjust based on available resources

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all candidate matching tasks, passing job_data and each candidate_data
        futures = [
            executor.submit(get_job_to_single_candidate_matching, job_data, candidate_data)
            for candidate_data in candidates_data
        ]
        
        # Process completed tasks as they finish
        for future in as_completed(futures):
            try:
                result = future.result()  # This will be the similarity result for one candidate
                results.append(result)  # Store the result for further processing if needed
            except Exception as e:
                print(f"Error processing candidate match: {e}")

    # Optionally return results or process them further
    return results

def get_job_to_single_candidate_matching(job_data: JobParsedDataDTO, candidate_data: CandidateParsedDataDTO):
    # Skills matching
    job_skills = job_data.skills
    candidate_skills = candidate_data.candidateSkills
    skills_matching_data = job_candidate_util.get_skills_matching(job_skills, candidate_skills, model_name="paraphrase-MiniLM-L3-v2")
    # print_pydantic_instance(skills_matching_data)

    # Qualification matching
    job_qualification = job_data.qualifications
    candidate_qualification = candidate_data.candidateQualifications
    qualifications_matching_data = job_candidate_util.get_tfidf_skills_matching2(job_qualification, candidate_qualification, threshold=0.4)
    # print_pydantic_instance(qualifications_matching_data)

    # Experience matching
    job_work_experience = job_data.experienceRole
    candidate_work_experience = candidate_data.candidateWorkExperiences
    ## Remove stop words and split into sentences. Use NLTK
    candidate_work_experience_processed = job_candidate_util.preprocess_text_sentences(candidate_work_experience)
    experience_matching_data = job_candidate_util.get_skills_matching(job_work_experience, candidate_work_experience_processed, model_name="paraphrase-MiniLM-L3-v2")
    print_pydantic_instance(experience_matching_data)

    return None


