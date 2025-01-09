from ..payload.request.job_to_candidates_matching_request_dto import (
    JobToCandidatesMatchingRequestDTO,
    CandidateToJobsMatchingRequestDTO,
    ParsedDataDTO,
)
from ..payload.response.general_matching_response import (
    MatchingListDTO,
    MatchingCriteriaDTO
)
from ..util import job_candidate_util
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_matching(
    type: str,
    job_to_candidates_matching_request_Dto: JobToCandidatesMatchingRequestDTO = None,
    candidate_to_jobs_matching_request_Dto: CandidateToJobsMatchingRequestDTO = None,
):
    # Check if the type is either 'candidates' or 'jobs'
    if (type == "candidates"):
        base_data = job_to_candidates_matching_request_Dto.jobData
        compare_datas = job_to_candidates_matching_request_Dto.candidatesData
    elif (type == "jobs"):
        base_data = candidate_to_jobs_matching_request_Dto.candidateData
        compare_datas = candidate_to_jobs_matching_request_Dto.jobsData
    else:
        raise ValueError("Invalid type. Must be either 'candidates' or 'jobs'")
    
    # To store results from each compare matching with their index
    results_with_indices = []

    # Define the maximum number of workers (threads)
    max_workers = min(10, len(compare_datas))  # Adjust based on available resources

    # Early exit if no compare_datas
    if (len(compare_datas) == 0):
        return MatchingListDTO(results = [])
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all compare matching tasks, passing base_data and each compare_data with its index
        futures = {
            executor.submit(get_base_to_compare_matching, base_data, compare_data): i
            for i, compare_data in enumerate(compare_datas)
        }
        # Process completed tasks as they finish
        for future in as_completed(futures):
            index = futures[future]  # Retrieve the original index
            try:
                result = future.result()  # This will be the similarity result for one compare
                results_with_indices.append((index, result))
            except Exception as e:
                print(f"Error processing candidate match at index {index}: {e}")

    # Sort results by the original index to maintain the order of compare_datas
    ordered_results = MatchingListDTO(
        results = [result for index, result in sorted(results_with_indices)]
    )

    # Optionally return results or process them further
    return ordered_results

def get_base_to_compare_matching(
    base_data: ParsedDataDTO, compare_data: ParsedDataDTO
):
    # Define each matching task as a separate function
    def match_job_title():
        base_job_title = base_data.jobTitle
        compare_job_title = compare_data.jobTitle
        return job_candidate_util.get_job_title_matching(base_job_title, compare_job_title)
    
    def match_skills():
        base_skills = base_data.skills
        compare_skills = compare_data.skills
        return job_candidate_util.get_skills_matching_v2(base_skills, compare_skills)

    def match_location():
        base_location = base_data.location
        compare_location = compare_data.location
        return job_candidate_util.get_location_matching(base_location, compare_location)

    def keyword_matching():
        base_keywords = base_data.keyWords
        compare_keywords = compare_data.keyWords
        return job_candidate_util.get_keyword_matching(base_keywords, compare_keywords)

    def match_qualifications():
        base_qualification = base_data.qualifications
        compare_qualification = compare_data.qualifications
        return job_candidate_util.get_qualification_matching(
            base_qualification, compare_qualification
        )

    def match_experience():
        base_work_experience = base_data.experienceRole
        compare_work_experience = compare_data.experienceRole
        return job_candidate_util.get_experience_matching(
            base_work_experience, compare_work_experience,
        )
    
    def match_years_of_experience():
        base_years_of_experience = base_data.yearsOfExperience
        compare_years_of_experience = compare_data.yearsOfExperience
        return job_candidate_util.get_years_experience_matching(
            base_years_of_experience, compare_years_of_experience
        )
    
    def match_salary():
        base_salary = base_data.salary
        compare_salary = compare_data.salary
        return job_candidate_util.get_salary_matching(base_salary, compare_salary)

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

        results = MatchingCriteriaDTO(**results_dict)
    
    # Calculate total score
    total_score = 0
    for (key, value) in results:
      if key != "total_score" :
        total_score += (value.overall_score / 8)
    results.total_score = total_score

    return results