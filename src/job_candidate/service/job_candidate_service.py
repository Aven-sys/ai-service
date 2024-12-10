from ..payload.request.job_to_candidates_matching_request_dto import (
    JobToCandidatesMatchingRequestDTO,
    JobParsedDataDTO,
    CandidateParsedDataDTO,
)
from ..util import job_candidate_util
from common.util.langchain_pydantic_model_generator import print_pydantic_instance
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

# NO OORDER
# def get_job_to_candidates_matching(
#     job_to_candidates_matching_request_Dto: JobToCandidatesMatchingRequestDTO,
# ):
#     job_data = job_to_candidates_matching_request_Dto.jobData
#     candidates_data = job_to_candidates_matching_request_Dto.candidatesData

#     # To store results from each candidate matching
#     results = []

#     # Define the maximum number of workers (threads)
#     max_workers = min(10, len(candidates_data))  # Adjust based on available resources

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         # Submit all candidate matching tasks, passing job_data and each candidate_data
#         futures = [
#             executor.submit(
#                 get_job_to_single_candidate_matching, job_data, candidate_data
#             )
#             for candidate_data in candidates_data
#         ]

#         # Process completed tasks as they finish
#         for future in as_completed(futures):
#             try:
#                 result = (
#                     future.result()
#                 )  # This will be the similarity result for one candidate
#                 results.append(
#                     result
#                 )  # Store the result for further processing if needed
#             except Exception as e:
#                 print(f"Error processing candidate match: {e}")

#     # Optionally return results or process them further
#     return results

def get_job_to_candidates_matching(
    job_to_candidates_matching_request_Dto: JobToCandidatesMatchingRequestDTO,
):
    job_data = job_to_candidates_matching_request_Dto.jobData
    candidates_data = job_to_candidates_matching_request_Dto.candidatesData

    # To store results from each candidate matching with their index
    results_with_indices = []

    # Define the maximum number of workers (threads)
    max_workers = min(10, len(candidates_data))  # Adjust based on available resources

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all candidate matching tasks, passing job_data and each candidate_data with its index
        futures = {
            executor.submit(get_job_to_single_candidate_matching, job_data, candidate_data): i
            for i, candidate_data in enumerate(candidates_data)
        }

        # Process completed tasks as they finish
        for future in as_completed(futures):
            index = futures[future]  # Retrieve the original index
            try:
                result = future.result()  # This will be the similarity result for one candidate
                results_with_indices.append((index, result))
            except Exception as e:
                print(f"Error processing candidate match at index {index}: {e}")

    # Sort results by the original index to maintain the order of candidates_data
    ordered_results = [result for index, result in sorted(results_with_indices)]
    print(ordered_results)
    # Optionally return results or process them further
    return ordered_results

def get_job_to_single_candidate_matching(
    job_data: JobParsedDataDTO, candidate_data: CandidateParsedDataDTO
):
    # Define each matching task as a separate function
    def match_skills():
        job_skills = job_data.skills
        candidate_skills = candidate_data.candidateSkills
        return job_candidate_util.get_skills_matching(
            job_skills, candidate_skills, model_name="paraphrase-MiniLM-L3-v2"
        )

    def match_qualifications():
        job_qualification = job_data.qualifications
        candidate_qualification = candidate_data.candidateQualifications
        return job_candidate_util.get_tfidf_skills_matching2(
            job_qualification, candidate_qualification, threshold=0.4
        )

    def match_experience():
        job_work_experience = job_data.experienceRole
        candidate_work_experience = candidate_data.candidateWorkExperiences
        candidate_work_experience_processed = (
            job_candidate_util.preprocess_text_sentences(candidate_work_experience)
        )
        return job_candidate_util.get_skills_matching(
            job_work_experience,
            candidate_work_experience_processed,
            model_name="paraphrase-MiniLM-L3-v2",
        )

    def match_job_title():
        job_work_experience = []
        if job_data.experienceRole:
            job_work_experience.extend(
                job_data.experienceRole
                if isinstance(job_data.experienceRole, list)
                else [job_data.experienceRole]
            )
        if job_data.jobTitle:
            job_work_experience.append(job_data.jobTitle)
        candidate_job_title = candidate_data.candidateJobTitles
        return job_candidate_util.get_skills_matching(
            job_work_experience,
            candidate_job_title,
            model_name="paraphrase-MiniLM-L3-v2",
        )

    def match_education():
        job_education = job_data.qualifications
        candidate_education = candidate_data.candidateEducation
        candidate_education_processed = job_candidate_util.preprocess_text_sentences(
            candidate_education
        )
        # return job_candidate_util.get_skills_matching(job_education, candidate_education_processed, model_name="paraphrase-MiniLM-L3-v2")
        return job_candidate_util.get_tfidf_skills_matching2(
            job_education, candidate_education_processed, threshold=0.4
        )
    
    def match_location():
        job_location = job_data.location
        candidate_location = candidate_data.candidateLocation
        return job_candidate_util.get_skills_matching(
            [job_location],
            [candidate_location],
            model_name="paraphrase-MiniLM-L3-v2",
        )

    def match_years_of_experience():
        job_years_of_experience = job_data.yearsOfExperience
        candidate_years_of_experience = candidate_data.candidateYearsOfExperience
        return job_candidate_util.get_years_of_experience_matching(
            job_years_of_experience, candidate_years_of_experience
        )
    
    def keyword_matching():
        job_keywords = job_data.keyWords
        candidate_keywords = candidate_data.candidateDetails
        candidate_keywords_processed = job_candidate_util.preprocess_text_words(candidate_keywords)
        return job_candidate_util.get_tfidf_skills_matching2(
            job_keywords, candidate_keywords_processed, threshold=0.4
        )

    # Run all tasks concurrently
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(match_skills): "skills",
            executor.submit(match_qualifications): "qualifications",
            executor.submit(match_experience): "experience",
            executor.submit(match_job_title): "job_title",
            executor.submit(match_education): "education",
            executor.submit(match_location): "location",
            executor.submit(match_years_of_experience): "years_of_experience",
            executor.submit(keyword_matching): "keywords",
        }

        results = {}
        for future in as_completed(futures):
            task_name = futures[future]
            try:
                results[task_name] = future.result()
            except Exception as exc:
                print(f"{task_name} matching generated an exception: {exc}")

    # Print or use the results
    print("=============================== Skills Matching ===============================")
    print_pydantic_instance(results.get('skills'))
    print()
    print("=============================== Qualification Matching ===============================")
    print_pydantic_instance(results.get('qualifications'))
    print()
    print("=============================== Experience/ Role Matching ===============================")
    print_pydantic_instance(results.get('experience'))
    print()
    print("=============================== Job Title Matching ===============================")
    print_pydantic_instance(results.get('job_title'))
    print()
    # print_pydantic_instance(results.get("education"))
    print("=============================== Location Matching ===============================")
    print_pydantic_instance(results.get("location"))
    print()
    print("=============================== Years of Experience Matching ===============================")
    print_pydantic_instance(results.get("years_of_experience"))
    print()
    print("=============================== Key Word Matching ===============================")
    print_pydantic_instance(results.get("keywords"))
    print()

    return results
