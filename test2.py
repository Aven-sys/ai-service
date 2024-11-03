from rank_bm25 import BM25Okapi
import re
import nltk
import math

def test_bm25_library():
    # Test corpus
    # corpus = [
    #     "This sample text is meant to test the similarity matcher functionality.",
    #     "This is when i say Hello world.",
    #     "Another completely different document about cats and dogs.",
    #     "This is a test document that has some similar words.",
    #     "Hello world this is a programming example."
    # ]
    
    # # Test queries
    # queries = [
    #     "This is a sample text for similarity matching.",
    #     "This is when i say Hello world.",
    #     "Something about cats",
    #     "test document similar"
    # ]

    corpus = ['Candidate Basic Information: Name: Jaheen Ahamed Current Position: Senior Cloud Engineer Current Location: Singapore Candidate Nationality: Singaporean Profile Summary: Seeking a challenging position involving technical and managerial skills in an environment where growth is possible. Primary Skill: AWS/Azure platforms, Infra as code using Terraform, Kubernetes, Container provisioning, OS level patching, Security concepts in hardening Primary Skills: AWS/Azure platforms, Infra as code using Terraform, Kubernetes, Container provisioning, OS level patching, Security concepts in hardening Skill 1: AWS/Azure platforms, Infra as code using Terraform, Kubernetes, Container provisioning, OS level patching, Security concepts in hardening Secondary Skills: Networking, TCP/IP, Routing Protocols, Documentation and Drawing of AWS Architecture, DevSecOps tooling and automation concepts, System programming & configuration, Technical application/architecture support, Active Directory, Database and script deployment, Architecture improvements, Security scanning and OS hardening, Proof of concept (POC) implementation, Administration level actions on Development, UAT, Staging and Production environments Candidate Education Details: Achieved BTech (Electronics Engineering) in Electrical Engineering from National University of Singapore (NUS), studied from 2021-01-01 to 2024-03-01. Achieved Diploma in Electronics in Electronics from Temasek Polytechnic, studied from 2014-01-01 to 2017-12-01. Achieved O Level from Changkat Changi Secondary School, studied from 2010-01-01 to 2013-12-01. Candidate Work Experience Details: Worked as Senior Cloud Engineer at Accenture Pte Ltd for 1 years 9 months. Role involved: Experience in architecting, designing, developing, and implementing cloud solutions on AWS/Azure platforms. Familiar with maintaining at scale enterprise online web architectures, microservices architectures. Able to design/build/run IaC tooling like terraform to automate environment build. Strong Experience and Working Knowledge working in AWS Cloud Environment such as VPC, Direct Connect, IAM, S3, EC2, EKS, Tagging, Cloud Monitoring, Config, CloudTrail, Lambda, ECS, CDN, Route53, Storage Gateway etc. Experience in Infra as code using Terraform, cloud formation, ansible and Shell scripts. Experience in Kubernetes and the various open-source security and monitoring technologies for Kubernetes e.g., Istio, Prometheus, Thanos etc. Experience in operation monitoring and has worked on ElasticSearch, Grafana, Fluentd. Experience in AWS Organization and Enterprise support. Experience in cost optimization with Reserved Instances, spot instances, cost explorer. Familiar with Networking, TCP/IP, Routing Protocols, Fiber Leased Lines Connectivity. Familiar with Linux/Windows based OS Provisioning, Maintenance and Support Operations. Familiar in Documentation and Drawing of AWS Architecture, Standards, and Operating Procedures. Consolidate account into AWS organisation. Define policies and standard for AWS organization. Familiar with the VPC architecture. Design transformation of on-prem connectivity to public cloud. Design and implement solution for multi-cloud interconnection architecture. Actively control and optimize AWS Cloud costs. Ensure Cloud environments are compliant to security policies. Perform scheduled maintenance activities for all related resources in AWS Cloud. Incident management and support during incidents and outages. Troubleshoot system and applications issues. Assess and Review AWS Cloud and Hybrid Environment for Optimisation and best practices. Perform SOP, Processes, Strategy and for AWS Cloud Environment. Management and monitoring of AWS Cloud Environment using relevant tools. Documentation and periodic reports on all AWS Cloud management, monitoring and maintenance activities. Document Cloud Inventory and Management. Demonstrating a strong desire and capability to learn, grow, and develop new skills for hybrid cloud. Familiar with container provisioning, images, repository lifecycle concepts & deployment of container centric services. Familiar with maintaining at enterprise Architectures including web application, Microservices. Familiar with DevSecOps tooling and automation concepts. Familiar with setting and defining operational processes. Familiar with OS level patching, Application-level patching, Operational housekeeping and archival, Backup and recovery etc. Familiar with gov level security concepts in hardening, scanning, security posture and mindset.. Familiar with Infra as Code using Ansible/Terraform Scripting. Worked as Infrastructure Engineer at NCS Pte Ltd for 8 months. Role involved: Plan, design, install, test & implement systems in accordance with specifications & service level. Where relevant, perform the necessary system programming & configuration. Perform and manage routine preventive maintenance and operational activities such as service requests, report generation, incident response, patch management. Perform Cloud infrastructure monitoring and escalation as per standard operations procedures. Assist the customer in solving issues related to Private Managed Cloud platforms.. Manage systems changes through established change request process & provide status reports to the relevant parties. Responsible for the system design, implementation, testing, managing and maintenance of proposed system. Support to users on a variety of system technologies including Microsoft Windows Server OS, Unix/Linux OS and CentOS. Provide technical application/architecture support to infra and application teams. Maintain and enhance application architecture components. Develop and maintain operation processes for Infrastructure deployment and maintenance. Generate regular report on the activities, performance to highlight the exceptions and fixes. Provide technical documentation support to Project & Solutions Teams, as well as hand over as-built system documentation to Maintenance support team. Manage and ensure timely patching deliverables conform to client patch management processes within the SLA given; define and provide test methodologies to test the patches; identify and manage risk and mitigation strategies; identify, analyses and ensure resolution of issues; manage change control. Scheduling, planning, resource allocation and prioritizing maintenance activities. Carry out monthly maintenance to back-up data and patch the Servers OS, monitor client performance measurement for client environments after every patching; and maintain records of the performance measurement taken. Provide 2nd line of support to address client related queries, incidents and requests reported on ground. Track system defects reported by users. Ensure prompt resolution is provided by the Contractors to the defects reported. Administer user, computer accounts, and group policies in the Active Directory to support technology exploration projects.. Work closely with Team Leader to meet system delivery milestones. Engage with back-line support and/or with vendor’s support to diagnose and rectify technical problems. Assist in day-to-day operation of the System Delivery team. Carry out other technical related duties that may be required. Support large scale IT infra environment. Maintain systems availability and performance. Worked as System Engineer at Xtremax Pte Ltd for 2 years. Role involved: Responsible for the setup, configuration, efficiency, hardening, deploying, automating, maintaining, managing and reliable operation of cloud, working AWS, Azure, Google Cloud. Ensure the uptime, performance, availability, scalability and security of the servers. Analyse, isolate and resolve both network and physical issues timely; make recommendations for future upgrades. Develop technical documentation and SOP procedures. Acquire, install, or upgrade computer components and software. Maintain backup including system backup, application backup, etc. Build, release, and configuration management of servers. Windows and Linux System Administration. Windows and Linux patch management. (Setup WSUS and Linux repo and push updates to clients). Database and script deployment for client sites. Pre-production Acceptance Testing to help assure the quality of our products/services. System troubleshooting and problem solving across platform and application domains. Suggesting architecture improvements, recommending process improvements. Security scanning and OS hardening. Ensuring critical system security through, the use of best-in-class cloud security solutions. Work with application and architecture teams to conduct proof of concept (POC) and implement the design in production environment. Fix deployment failures and find out root cause for the issues. Perform all kind of administration level actions on Development, UAT, Staging and Production environments. Worked as Associate Engineer at AFPD Pte Ltd for 1 years 6 months. Role involved: Perform equipment periodic maintenance activities. Assist Engineer on machine improvement and parts evaluation. Setup equipment maintenance SOP and its work procedure. Repair and troubleshoot on machine breakdown. Manage spare parts inventory. Assist in any other related duties assigned by immediate supervisor as when necessary. Total Work Experience: 5 years and 11 months.']
    
    # Test queries
    queries = ['Job Title: Docker Kub Country: American Samoa Job Type: Permanent Qualification: Bachelorssss Languages requirement job: Chinese Job Description: seeking highly motivated skilled Docker Engineer manage optimize containerized applications infrastructure. ideal candidate deep experience Docker, container orchestration tools like Kubernetes, CI/CD pipelines. collaborate closely development operations teams ensure efficient containerization, deployment, scaling applications. Key Responsibilities: Containerization: Develop maintain Docker containers various applications microservices. Docker Management: Manage Docker images, containers, registries, ensuring scalability security. Orchestration: Implement manage container orchestration tools Kubernetes, Docker Swarm, OpenShift. CI/CD Pipelines: Integrate Docker containers CI/CD pipelines using tools like Jenkins, GitLab CI, Azure DevOps. Monitoring & Logging: Set monitoring logging solutions container environments (e.g., Prometheus, Grafana, ELK). Security: Implement security best practices containers, images, registries (e.g., vulnerability scanning). Performance Optimization: Analyze fine-tune Docker infrastructure optimal performance resource utilization. Collaboration: Work developers optimize applications containerization. Troubleshooting: Resolve container-related issues provide support production systems. Documentation: Create maintain technical documentation related container workflows standards. Skills & Qualifications: Strong experience Docker containerization applications. Hands-on experience container orchestration tools like Kubernetes, Docker Swarm, OpenShift. Expertise CI/CD tools Jenkins, GitLab CI, Azure DevOps. Knowledge Linux operating systems shell scripting. Experience cloud platforms (AWS, Azure, GCP) deploying containers cloud environments. Strong understanding networking concepts related containers (e.g., overlay networks). Familiarity security practices containers registries (e.g., image scanning tools like Trivy). Knowledge monitoring logging tools like Prometheus, Grafana, ELK Stack. Ability troubleshoot complex production issues related container environments. Excellent problem-solving skills ability work fast-paced environment. Education & Experience: Bachelor''s degree Computer Science, Engineering, related field (or equivalent experience). 3+ years experience working Docker containerized applications. Certification Docker, Kubernetes, cloud platforms (e.g., CKA, Docker Certified Associate) plus. Preferred Qualifications: Experience Infrastructure Code (IaC) tools like Terraform Ansible. Familiarity microservices architecture best practices. Knowledge serverless technologies hybrid cloud deployments. Hands-on experience GitOps methodologies. Offer: Competitive salary benefits package. Opportunity work cutting-edge containerization technologies. Flexible work environment remote options. Access training professional development resources.']
    
    # Tokenize corpus
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    
    # Initialize BM25 with adjusted parameters
    bm25 = BM25Okapi(
        tokenized_corpus,
        # k1=1.2,  # Default is 1.5
        # b=0.35,   # Default is 0.75
        # epsilon=0.25  # Smoothing parameter
    )
    
    # Test each query
    print("Testing BM25 Library Scoring:\n")
    for query in queries:
        print(f"\nQuery: {query}")
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get scores
        scores = bm25.get_scores(tokenized_query)
        
        # Print scores with documents
        print("\nMatches:")
        scored_docs = list(zip(corpus, scores))
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        for doc, score in scored_docs:
            print(f"\nDocument: {doc}")
            print(f"Score: {score:.4f}")
        
        print("\n" + "="*80)

def preprocess_text(text):
    # Lowercase, remove special characters, and split by whitespace
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()).split()

def split_into_segments(text, segment_size=50):
    # Split text into tokens, then group into segments of a certain size
    tokens = preprocess_text(text)
    return [tokens[i:i + segment_size] for i in range(0, len(tokens), segment_size)]

def compare_documents_with_segments(doc1, doc2):
    # Split both documents into smaller segments
    doc1_segments = split_into_segments(doc1)
    doc2_segments = split_into_segments(doc2)

    # Create a BM25 corpus with segments from both documents
    bm25 = BM25Okapi(doc1_segments)

    # Score each segment in doc2 against doc1 corpus
    scores = []
    for query_segment in doc2_segments:
        segment_score = bm25.get_scores(query_segment)
        scores.append(max(segment_score))  # Take the highest score for each query segment

    # Calculate the average score across all segments
    average_score = sum(scores) / len(scores) if scores else 0
    return average_score

def split_into_sentences(text):
    # Split text into sentences and preprocess each sentence
    sentences = nltk.sent_tokenize(text)
    return [preprocess_text(sentence) for sentence in sentences]

def compare_documents_with_sentence_bm25(doc1, doc2):
    # Split both documents into sentences
    doc1_sentences = split_into_sentences(doc1)
    doc2_sentences = split_into_sentences(doc2)

    # Create a BM25 corpus with sentences from doc1
    bm25 = BM25Okapi(doc1_sentences)

    # Score each sentence in doc2 against the sentences in doc1 corpus
    scores = []
    for query_sentence in doc2_sentences:
        sentence_scores = bm25.get_scores(query_sentence)
        scores.append(max(sentence_scores))  # Take the highest score for each query sentence

    # Calculate the average score across all sentences
    average_score = sum(scores) / len(scores) if scores else 0
    return average_score

def sigmoid(x, alpha=1.0):
    return 1 / (1 + math.exp(-alpha * x))

if __name__ == "__main__":
    # test_bm25_library()
    # corpus = ['Candidate Basic Information: Name: Jaheen Ahamed Current Position: Senior Cloud Engineer Current Location: Singapore Candidate Nationality: Singaporean Profile Summary: Seeking a challenging position involving technical and managerial skills in an environment where growth is possible. Primary Skill: AWS/Azure platforms, Infra as code using Terraform, Kubernetes, Container provisioning, OS level patching, Security concepts in hardening Primary Skills: AWS/Azure platforms, Infra as code using Terraform, Kubernetes, Container provisioning, OS level patching, Security concepts in hardening Skill 1: AWS/Azure platforms, Infra as code using Terraform, Kubernetes, Container provisioning, OS level patching, Security concepts in hardening Secondary Skills: Networking, TCP/IP, Routing Protocols, Documentation and Drawing of AWS Architecture, DevSecOps tooling and automation concepts, System programming & configuration, Technical application/architecture support, Active Directory, Database and script deployment, Architecture improvements, Security scanning and OS hardening, Proof of concept (POC) implementation, Administration level actions on Development, UAT, Staging and Production environments Candidate Education Details: Achieved BTech (Electronics Engineering) in Electrical Engineering from National University of Singapore (NUS), studied from 2021-01-01 to 2024-03-01. Achieved Diploma in Electronics in Electronics from Temasek Polytechnic, studied from 2014-01-01 to 2017-12-01. Achieved O Level from Changkat Changi Secondary School, studied from 2010-01-01 to 2013-12-01. Candidate Work Experience Details: Worked as Senior Cloud Engineer at Accenture Pte Ltd for 1 years 9 months. Role involved: Experience in architecting, designing, developing, and implementing cloud solutions on AWS/Azure platforms. Familiar with maintaining at scale enterprise online web architectures, microservices architectures. Able to design/build/run IaC tooling like terraform to automate environment build. Strong Experience and Working Knowledge working in AWS Cloud Environment such as VPC, Direct Connect, IAM, S3, EC2, EKS, Tagging, Cloud Monitoring, Config, CloudTrail, Lambda, ECS, CDN, Route53, Storage Gateway etc. Experience in Infra as code using Terraform, cloud formation, ansible and Shell scripts. Experience in Kubernetes and the various open-source security and monitoring technologies for Kubernetes e.g., Istio, Prometheus, Thanos etc. Experience in operation monitoring and has worked on ElasticSearch, Grafana, Fluentd. Experience in AWS Organization and Enterprise support. Experience in cost optimization with Reserved Instances, spot instances, cost explorer. Familiar with Networking, TCP/IP, Routing Protocols, Fiber Leased Lines Connectivity. Familiar with Linux/Windows based OS Provisioning, Maintenance and Support Operations. Familiar in Documentation and Drawing of AWS Architecture, Standards, and Operating Procedures. Consolidate account into AWS organisation. Define policies and standard for AWS organization. Familiar with the VPC architecture. Design transformation of on-prem connectivity to public cloud. Design and implement solution for multi-cloud interconnection architecture. Actively control and optimize AWS Cloud costs. Ensure Cloud environments are compliant to security policies. Perform scheduled maintenance activities for all related resources in AWS Cloud. Incident management and support during incidents and outages. Troubleshoot system and applications issues. Assess and Review AWS Cloud and Hybrid Environment for Optimisation and best practices. Perform SOP, Processes, Strategy and for AWS Cloud Environment. Management and monitoring of AWS Cloud Environment using relevant tools. Documentation and periodic reports on all AWS Cloud management, monitoring and maintenance activities. Document Cloud Inventory and Management. Demonstrating a strong desire and capability to learn, grow, and develop new skills for hybrid cloud. Familiar with container provisioning, images, repository lifecycle concepts & deployment of container centric services. Familiar with maintaining at enterprise Architectures including web application, Microservices. Familiar with DevSecOps tooling and automation concepts. Familiar with setting and defining operational processes. Familiar with OS level patching, Application-level patching, Operational housekeeping and archival, Backup and recovery etc. Familiar with gov level security concepts in hardening, scanning, security posture and mindset.. Familiar with Infra as Code using Ansible/Terraform Scripting. Worked as Infrastructure Engineer at NCS Pte Ltd for 8 months. Role involved: Plan, design, install, test & implement systems in accordance with specifications & service level. Where relevant, perform the necessary system programming & configuration. Perform and manage routine preventive maintenance and operational activities such as service requests, report generation, incident response, patch management. Perform Cloud infrastructure monitoring and escalation as per standard operations procedures. Assist the customer in solving issues related to Private Managed Cloud platforms.. Manage systems changes through established change request process & provide status reports to the relevant parties. Responsible for the system design, implementation, testing, managing and maintenance of proposed system. Support to users on a variety of system technologies including Microsoft Windows Server OS, Unix/Linux OS and CentOS. Provide technical application/architecture support to infra and application teams. Maintain and enhance application architecture components. Develop and maintain operation processes for Infrastructure deployment and maintenance. Generate regular report on the activities, performance to highlight the exceptions and fixes. Provide technical documentation support to Project & Solutions Teams, as well as hand over as-built system documentation to Maintenance support team. Manage and ensure timely patching deliverables conform to client patch management processes within the SLA given; define and provide test methodologies to test the patches; identify and manage risk and mitigation strategies; identify, analyses and ensure resolution of issues; manage change control. Scheduling, planning, resource allocation and prioritizing maintenance activities. Carry out monthly maintenance to back-up data and patch the Servers OS, monitor client performance measurement for client environments after every patching; and maintain records of the performance measurement taken. Provide 2nd line of support to address client related queries, incidents and requests reported on ground. Track system defects reported by users. Ensure prompt resolution is provided by the Contractors to the defects reported. Administer user, computer accounts, and group policies in the Active Directory to support technology exploration projects.. Work closely with Team Leader to meet system delivery milestones. Engage with back-line support and/or with vendor’s support to diagnose and rectify technical problems. Assist in day-to-day operation of the System Delivery team. Carry out other technical related duties that may be required. Support large scale IT infra environment. Maintain systems availability and performance. Worked as System Engineer at Xtremax Pte Ltd for 2 years. Role involved: Responsible for the setup, configuration, efficiency, hardening, deploying, automating, maintaining, managing and reliable operation of cloud, working AWS, Azure, Google Cloud. Ensure the uptime, performance, availability, scalability and security of the servers. Analyse, isolate and resolve both network and physical issues timely; make recommendations for future upgrades. Develop technical documentation and SOP procedures. Acquire, install, or upgrade computer components and software. Maintain backup including system backup, application backup, etc. Build, release, and configuration management of servers. Windows and Linux System Administration. Windows and Linux patch management. (Setup WSUS and Linux repo and push updates to clients). Database and script deployment for client sites. Pre-production Acceptance Testing to help assure the quality of our products/services. System troubleshooting and problem solving across platform and application domains. Suggesting architecture improvements, recommending process improvements. Security scanning and OS hardening. Ensuring critical system security through, the use of best-in-class cloud security solutions. Work with application and architecture teams to conduct proof of concept (POC) and implement the design in production environment. Fix deployment failures and find out root cause for the issues. Perform all kind of administration level actions on Development, UAT, Staging and Production environments. Worked as Associate Engineer at AFPD Pte Ltd for 1 years 6 months. Role involved: Perform equipment periodic maintenance activities. Assist Engineer on machine improvement and parts evaluation. Setup equipment maintenance SOP and its work procedure. Repair and troubleshoot on machine breakdown. Manage spare parts inventory. Assist in any other related duties assigned by immediate supervisor as when necessary. Total Work Experience: 5 years and 11 months.']
    # corpus =["Jane Doe. Dedicated and enthusiastic teacher with over 5 years of experience in fostering a positive and inclusive classroom environment. Skilled in curriculum development, lesson planning, and implementing innovative teaching techniques to accommodate diverse student needs. Passionate about helping students reach their full potential by promoting active learning and critical thinking. Education: Master of Education (M.Ed.) in Curriculum and Instruction, University of Education, City, State, Graduated: May 2020; Bachelor of Arts in English Education, State College, City, State, Graduated: May 2018. Professional Experience: English Teacher, City High School, City, State, August 2020 – Present - Teach English Language Arts to grades 9-12, designing engaging lessons that meet state standards and enhance critical thinking and literacy skills. Develop and adapt curriculum to accommodate diverse learning styles, creating a supportive and inclusive classroom environment. Assess student progress through formative and summative evaluations, providing regular feedback to students and parents. Incorporate technology and digital resources to foster interactive learning, resulting in a 20% increase in student engagement and participation. Serve as the advisor for the school’s journalism club, guiding students in publishing the school newspaper and developing their writing skills. Substitute Teacher, District School System, City, State, September 2018 – July 2020 - Covered various subjects across K-12, effectively managing classrooms and delivering prepared lessons. Adapted quickly to different classroom settings, ensuring a positive and productive learning environment for students of varying backgrounds and academic levels. Maintained clear communication with teachers and administrators, reporting on classroom activities and student progress. Skills: Curriculum Design & Lesson Planning - Experienced in creating interactive lessons that align with Common Core standards and encourage student participation; Classroom Management - Skilled at maintaining a structured and supportive classroom environment"]
    # corpus =["Highly skilled DevOps Engineer with over 6 years of experience in implementing CI/CD pipelines, managing cloud infrastructure, and optimizing application deployment with Docker and Kubernetes. Proven ability to improve system reliability and streamline software delivery processes. Adept at collaborating with cross-functional teams to deliver high-quality solutions in fast-paced environments. Education: Bachelor of Science in Computer Science, Tech University, City, State, Graduated: May 2015. Professional Experience: DevOps Engineer, Tech Solutions Inc., City, State, June 2018 – Present - Designed and implemented automated CI/CD pipelines using Jenkins, reducing deployment time by 30%. Developed and managed containerized applications using Docker, orchestrated with Kubernetes for efficient scaling and resource utilization. Configured and monitored AWS infrastructure, including EC2, RDS, and S3, optimizing costs by 15%. Collaborated closely with developers to troubleshoot issues and optimize application performance. Created robust monitoring and logging solutions with Prometheus and Grafana to enhance system reliability. DevOps Specialist, Cloud Innovators LLC, City, State, September 2015 – May 2018 - Built and maintained automated build and deployment processes, ensuring rapid and reliable releases. Utilized Docker to containerize applications, creating standardized environments for development and testing. Set up continuous integration workflows with GitLab CI, enhancing code quality and reducing bugs. Automated server provisioning and configuration management with Ansible, saving time on manual processes. Skills: CI/CD Pipelines - Experienced with Jenkins, GitLab CI, and GitHub Actions for end-to-end automation; Docker & Kubernetes - Expert in containerization and orchestration; Cloud Management - Proficient in AWS, Azure; Configuration Management - Skilled in Ansible, Terraform for infrastructure as code; Monitoring & Logging - Knowledgeable in Prometheus, Grafana, and ELK Stack for system health tracking"]
    corpus = ['Experienced financial analyst with a strong background in accounting, data analysis, and reporting. Proficient in financial modeling, budgeting, and forecasting. Skilled in tools like Excel, SAP, and Power BI to analyze and visualize financial data. Demonstrated ability to work in high-pressure environments, providing insights to guide business decisions. Excellent communication skills and experience collaborating with stakeholders to deliver financial reports and strategy recommendations.']

    # Test queries
    queries = ['Job Title: Docker hub Country: American Samoa Job Type: Permanent Qualification: Bachelorssss Languages requirement job: Chinese Job Description: seeking highly motivated skilled Docker Engineer manage optimize containerized applications infrastructure. ideal candidate deep experience Docker, container orchestration tools like Kubernetes, CI/CD pipelines. collaborate closely development operations teams ensure efficient containerization, deployment, scaling applications. Key Responsibilities: Containerization: Develop maintain Docker containers various applications microservices. Docker Management: Manage Docker images, containers, registries, ensuring scalability security. Orchestration: Implement manage container orchestration tools Kubernetes, Docker Swarm, OpenShift. CI/CD Pipelines: Integrate Docker containers CI/CD pipelines using tools like Jenkins, GitLab CI, Azure DevOps. Monitoring & Logging: Set monitoring logging solutions container environments (e.g., Prometheus, Grafana, ELK). Security: Implement security best practices containers, images, registries (e.g., vulnerability scanning). Performance Optimization: Analyze fine-tune Docker infrastructure optimal performance resource utilization. Collaboration: Work developers optimize applications containerization. Troubleshooting: Resolve container-related issues provide support production systems. Documentation: Create maintain technical documentation related container workflows standards. Skills & Qualifications: Strong experience Docker containerization applications. Hands-on experience container orchestration tools like Kubernetes, Docker Swarm, OpenShift. Expertise CI/CD tools Jenkins, GitLab CI, Azure DevOps. Knowledge Linux operating systems shell scripting. Experience cloud platforms (AWS, Azure, GCP) deploying containers cloud environments. Strong understanding networking concepts related containers (e.g., overlay networks). Familiarity security practices containers registries (e.g., image scanning tools like Trivy). Knowledge monitoring logging tools like Prometheus, Grafana, ELK Stack. Ability troubleshoot complex production issues related container environments. Excellent problem-solving skills ability work fast-paced environment. Education & Experience: Bachelor''s degree Computer Science, Engineering, related field (or equivalent experience). 3+ years experience working Docker containerized applications. Certification Docker, Kubernetes, cloud platforms (e.g., CKA, Docker Certified Associate) plus. Preferred Qualifications: Experience Infrastructure Code (IaC) tools like Terraform Ansible. Familiarity microservices architecture best practices. Knowledge serverless technologies hybrid cloud deployments. Hands-on experience GitOps methodologies. Offer: Competitive salary benefits package. Opportunity work cutting-edge containerization technologies. Flexible work environment remote options. Access training professional development resources.']

    # queries = ['Experienced financial analyst with a strong background in accounting, data analysis, and reporting. Proficient in financial modeling, budgeting, and forecasting. Skilled in tools like Excel, SAP, and Power BI to analyze and visualize financial data. Demonstrated ability to work in high-pressure environments, providing insights to guide business decisions. Excellent communication skills and experience collaborating with stakeholders to deliver financial reports and strategy recommendations.']
    # queries = ['Hello world. What are you guys doing tonight? Any one up for some dinner? Hello world. What are you guys doing tonight? Any one up for some dinner? Hello world. What are you guys doing tonight? Any one up for some dinner? Hello world. What are you guys doing tonight? Any one up for some dinner? Hello world. What are you guys doing tonight? Any one up for some dinner? Hello world. What are you guys doing tonight? Experienced financial analyst with a strong background in accounting. ']

    score = compare_documents_with_sentence_bm25(corpus[0], queries[0])
    print(score)
    # print("Normalized Score: ", sigmoid(score, alpha=1))
