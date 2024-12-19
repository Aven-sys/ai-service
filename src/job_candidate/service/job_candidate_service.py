from ..payload.response.general_matching_response import (
    JobToCandidateMatchingDTO, 
    JobToCandidateMatchingListDTO,
    CandidateToJobMatchingListDTO
)
from ..payload.request.job_to_candidates_matching_request_dto import (
    JobToCandidatesMatchingRequestDTO,
    JobToCandidatesMatchingRequestV2DTO,
    CandidateToJobsMatchingRequestDTO,
    JobParsedDataDTO,
    JobParsedDataV2DTO,
    CandidateParsedDataDTO,
    CandidateParsedDataV2DTO,
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

def get_job_to_candidates_matching_v3(
    job_to_candidates_matching_request_Dto: JobToCandidatesMatchingRequestV2DTO,
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
            executor.submit(get_job_to_single_candidate_matching_v3, job_data, candidate_data): i
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
    ordered_results = JobToCandidateMatchingListDTO(
        candidates = [result for index, result in sorted(results_with_indices)]
    )

    # Optionally return results or process them further
    return ordered_results

def get_job_to_single_candidate_matching_v3(
    job_data: JobParsedDataDTO, candidate_data: CandidateParsedDataV2DTO
):
    # Define each matching task as a separate function
    def match_job_title():
      job_title = [job_data.jobTitle]
      candidate_job_title = candidate_data.jobTitle
      return job_candidate_util.get_job_title_matching(
          job_title,
          candidate_job_title,
      )
    
    def match_skills():
        job_skills = job_data.skills
        candidate_skills = candidate_data.skills
        return job_candidate_util.get_skills_matching_v2(
            job_skills, candidate_skills
        )

    def match_location():
        job_location = job_data.location
        candidate_location = candidate_data.location
        return job_candidate_util.get_location_matching(
            [job_location],
            [candidate_location],
        )

    def keyword_matching():
        job_keywords = job_data.keyWords
        candidate_keywords = candidate_data.keyWords
        return job_candidate_util.get_keyword_matching(
            job_keywords, candidate_keywords
        )

    def match_qualifications():
        job_qualification = job_data.qualifications
        candidate_qualification = candidate_data.qualifications
        return job_candidate_util.get_qualification_matching(
            job_qualification, candidate_qualification
        )

    def match_experience():
        job_work_experience = job_data.experienceRole
        candidate_work_experience = candidate_data.experienceRole
        return job_candidate_util.get_experience_matching(
            job_work_experience,
            candidate_work_experience,
        )
    
    def match_years_of_experience():
        job_years_of_experience = job_data.yearsOfExperience
        candidate_years_of_experience = candidate_data.yearsOfExperience
        return job_candidate_util.get_years_experience_matching(
            job_years_of_experience, candidate_years_of_experience
        )
    
    def match_salary():
        job_salary = job_data.salary
        candidate_salary = candidate_data.salary
        return job_candidate_util.get_salary_matching(job_salary, candidate_salary)

    # Run all tasks concurrently
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(match_job_title): "job_title",
            executor.submit(match_skills): "skills",
            executor.submit(match_location): "location",
            executor.submit(keyword_matching): "keywords",
            executor.submit(match_salary): "salary",
            executor.submit(match_years_of_experience): "years_of_experience",
            executor.submit(match_experience): "experience",
            executor.submit(match_qualifications): "qualifications",
        }

        results_dict = {}
        for future in as_completed(futures):
            task_name = futures[future]
            try:
                results_dict[task_name] = future.result()
            except Exception as exc:
                print(f"{task_name} matching generated an exception: {exc}")

        results = JobToCandidateMatchingDTO(**results_dict)
    
    # Calculate total score
    total_score = 0
    for (key, value) in results:
      if key != "total_score" :
        total_score += (value.overall_score / 8)
    results.total_score = total_score

    return results

def get_candidate_to_jobs_matching(
    candidate_to_jobs_matching_request_Dto: CandidateToJobsMatchingRequestDTO,
):
    jobs_data = candidate_to_jobs_matching_request_Dto.jobsData
    candidate_data = candidate_to_jobs_matching_request_Dto.candidateData

    # To store results from each candidate matching with their index
    results_with_indices = []

    # Define the maximum number of workers (threads)
    max_workers = min(10, len(jobs_data))  # Adjust based on available resources

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all candidate matching tasks, passing job_data and each candidate_data with its index
        futures = {
            executor.submit(get_candidate_to_single_job_matching, candidate_data, job_data): i
            for i, job_data in enumerate(jobs_data)
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
    ordered_results = CandidateToJobMatchingListDTO(
        jobs = [result for index, result in sorted(results_with_indices)]
    )
    # Optionally return results or process them further
    return ordered_results

def get_candidate_to_single_job_matching(
    candidate_data: CandidateParsedDataV2DTO, job_data: JobParsedDataV2DTO
):
    # Define each matching task as a separate function
    def match_job_title():
      job_title = [job_data.jobTitle]
      candidate_job_title = candidate_data.jobTitle
      return job_candidate_util.get_job_title_matching(
          job_title,
          candidate_job_title,
      )
    
    def match_skills():
        job_skills = job_data.skills
        candidate_skills = candidate_data.skills
        return job_candidate_util.get_skills_matching_v2(
            job_skills, candidate_skills
        )

    def match_location():
        job_location = job_data.location
        candidate_location = candidate_data.location
        return job_candidate_util.get_location_matching(
            [job_location],
            [candidate_location],
        )

    def keyword_matching():
        job_keywords = job_data.keyWords
        candidate_keywords = candidate_data.keyWords
        return job_candidate_util.get_keyword_matching(
            job_keywords, candidate_keywords
        )

    def match_qualifications():
        job_qualification = job_data.qualifications
        candidate_qualification = candidate_data.qualifications
        return job_candidate_util.get_qualification_matching(
            job_qualification, candidate_qualification
        )

    def match_experience():
        job_work_experience = job_data.experienceRole
        candidate_work_experience = candidate_data.experienceRole
        return job_candidate_util.get_experience_matching(
            job_work_experience,
            candidate_work_experience,
        )
    
    def match_years_of_experience():
        job_years_of_experience = job_data.yearsOfExperience
        candidate_years_of_experience = candidate_data.yearsOfExperience
        return job_candidate_util.get_years_experience_matching(
            job_years_of_experience, candidate_years_of_experience
        )
    
    def match_salary():
        job_salary = job_data.salary
        candidate_salary = candidate_data.salary
        return job_candidate_util.get_salary_matching(job_salary, candidate_salary)

    # Run all tasks concurrently
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(match_job_title): "job_title",
            executor.submit(match_skills): "skills",
            executor.submit(match_location): "location",
            executor.submit(keyword_matching): "keywords",
            executor.submit(match_salary): "salary",
            executor.submit(match_years_of_experience): "years_of_experience",
            executor.submit(match_experience): "experience",
            executor.submit(match_qualifications): "qualifications",
        }

        results_dict = {}
        for future in as_completed(futures):
            task_name = futures[future]
            try:
                results_dict[task_name] = future.result()
            except Exception as exc:
                print(f"{task_name} matching generated an exception: {exc}")

        results = JobToCandidateMatchingDTO(**results_dict)
    
    # Calculate total score
    total_score = 0
    for (key, value) in results:
      if key != "total_score" :
        total_score += (value.overall_score / 8)
    results.total_score = total_score

    return results