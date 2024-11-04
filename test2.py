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

    # corpus = ['Candidate Basic Information: Name: Jaheen Ahamed Current Position: Senior Cloud Engineer Current Location: Singapore Candidate Nationality: Singaporean Profile Summary: Seeking a challenging position involving technical and managerial skills in an environment where growth is possible. Primary Skill: AWS/Azure platforms, Infra as code using Terraform, Kubernetes, Container provisioning, OS level patching, Security concepts in hardening Primary Skills: AWS/Azure platforms, Infra as code using Terraform, Kubernetes, Container provisioning, OS level patching, Security concepts in hardening Skill 1: AWS/Azure platforms, Infra as code using Terraform, Kubernetes, Container provisioning, OS level patching, Security concepts in hardening Secondary Skills: Networking, TCP/IP, Routing Protocols, Documentation and Drawing of AWS Architecture, DevSecOps tooling and automation concepts, System programming & configuration, Technical application/architecture support, Active Directory, Database and script deployment, Architecture improvements, Security scanning and OS hardening, Proof of concept (POC) implementation, Administration level actions on Development, UAT, Staging and Production environments Candidate Education Details: Achieved BTech (Electronics Engineering) in Electrical Engineering from National University of Singapore (NUS), studied from 2021-01-01 to 2024-03-01. Achieved Diploma in Electronics in Electronics from Temasek Polytechnic, studied from 2014-01-01 to 2017-12-01. Achieved O Level from Changkat Changi Secondary School, studied from 2010-01-01 to 2013-12-01. Candidate Work Experience Details: Worked as Senior Cloud Engineer at Accenture Pte Ltd for 1 years 9 months. Role involved: Experience in architecting, designing, developing, and implementing cloud solutions on AWS/Azure platforms. Familiar with maintaining at scale enterprise online web architectures, microservices architectures. Able to design/build/run IaC tooling like terraform to automate environment build. Strong Experience and Working Knowledge working in AWS Cloud Environment such as VPC, Direct Connect, IAM, S3, EC2, EKS, Tagging, Cloud Monitoring, Config, CloudTrail, Lambda, ECS, CDN, Route53, Storage Gateway etc. Experience in Infra as code using Terraform, cloud formation, ansible and Shell scripts. Experience in Kubernetes and the various open-source security and monitoring technologies for Kubernetes e.g., Istio, Prometheus, Thanos etc. Experience in operation monitoring and has worked on ElasticSearch, Grafana, Fluentd. Experience in AWS Organization and Enterprise support. Experience in cost optimization with Reserved Instances, spot instances, cost explorer. Familiar with Networking, TCP/IP, Routing Protocols, Fiber Leased Lines Connectivity. Familiar with Linux/Windows based OS Provisioning, Maintenance and Support Operations. Familiar in Documentation and Drawing of AWS Architecture, Standards, and Operating Procedures. Consolidate account into AWS organisation. Define policies and standard for AWS organization. Familiar with the VPC architecture. Design transformation of on-prem connectivity to public cloud. Design and implement solution for multi-cloud interconnection architecture. Actively control and optimize AWS Cloud costs. Ensure Cloud environments are compliant to security policies. Perform scheduled maintenance activities for all related resources in AWS Cloud. Incident management and support during incidents and outages. Troubleshoot system and applications issues. Assess and Review AWS Cloud and Hybrid Environment for Optimisation and best practices. Perform SOP, Processes, Strategy and for AWS Cloud Environment. Management and monitoring of AWS Cloud Environment using relevant tools. Documentation and periodic reports on all AWS Cloud management, monitoring and maintenance activities. Document Cloud Inventory and Management. Demonstrating a strong desire and capability to learn, grow, and develop new skills for hybrid cloud. Familiar with container provisioning, images, repository lifecycle concepts & deployment of container centric services. Familiar with maintaining at enterprise Architectures including web application, Microservices. Familiar with DevSecOps tooling and automation concepts. Familiar with setting and defining operational processes. Familiar with OS level patching, Application-level patching, Operational housekeeping and archival, Backup and recovery etc. Familiar with gov level security concepts in hardening, scanning, security posture and mindset.. Familiar with Infra as Code using Ansible/Terraform Scripting. Worked as Infrastructure Engineer at NCS Pte Ltd for 8 months. Role involved: Plan, design, install, test & implement systems in accordance with specifications & service level. Where relevant, perform the necessary system programming & configuration. Perform and manage routine preventive maintenance and operational activities such as service requests, report generation, incident response, patch management. Perform Cloud infrastructure monitoring and escalation as per standard operations procedures. Assist the customer in solving issues related to Private Managed Cloud platforms.. Manage systems changes through established change request process & provide status reports to the relevant parties. Responsible for the system design, implementation, testing, managing and maintenance of proposed system. Support to users on a variety of system technologies including Microsoft Windows Server OS, Unix/Linux OS and CentOS. Provide technical application/architecture support to infra and application teams. Maintain and enhance application architecture components. Develop and maintain operation processes for Infrastructure deployment and maintenance. Generate regular report on the activities, performance to highlight the exceptions and fixes. Provide technical documentation support to Project & Solutions Teams, as well as hand over as-built system documentation to Maintenance support team. Manage and ensure timely patching deliverables conform to client patch management processes within the SLA given; define and provide test methodologies to test the patches; identify and manage risk and mitigation strategies; identify, analyses and ensure resolution of issues; manage change control. Scheduling, planning, resource allocation and prioritizing maintenance activities. Carry out monthly maintenance to back-up data and patch the Servers OS, monitor client performance measurement for client environments after every patching; and maintain records of the performance measurement taken. Provide 2nd line of support to address client related queries, incidents and requests reported on ground. Track system defects reported by users. Ensure prompt resolution is provided by the Contractors to the defects reported. Administer user, computer accounts, and group policies in the Active Directory to support technology exploration projects.. Work closely with Team Leader to meet system delivery milestones. Engage with back-line support and/or with vendor’s support to diagnose and rectify technical problems. Assist in day-to-day operation of the System Delivery team. Carry out other technical related duties that may be required. Support large scale IT infra environment. Maintain systems availability and performance. Worked as System Engineer at Xtremax Pte Ltd for 2 years. Role involved: Responsible for the setup, configuration, efficiency, hardening, deploying, automating, maintaining, managing and reliable operation of cloud, working AWS, Azure, Google Cloud. Ensure the uptime, performance, availability, scalability and security of the servers. Analyse, isolate and resolve both network and physical issues timely; make recommendations for future upgrades. Develop technical documentation and SOP procedures. Acquire, install, or upgrade computer components and software. Maintain backup including system backup, application backup, etc. Build, release, and configuration management of servers. Windows and Linux System Administration. Windows and Linux patch management. (Setup WSUS and Linux repo and push updates to clients). Database and script deployment for client sites. Pre-production Acceptance Testing to help assure the quality of our products/services. System troubleshooting and problem solving across platform and application domains. Suggesting architecture improvements, recommending process improvements. Security scanning and OS hardening. Ensuring critical system security through, the use of best-in-class cloud security solutions. Work with application and architecture teams to conduct proof of concept (POC) and implement the design in production environment. Fix deployment failures and find out root cause for the issues. Perform all kind of administration level actions on Development, UAT, Staging and Production environments. Worked as Associate Engineer at AFPD Pte Ltd for 1 years 6 months. Role involved: Perform equipment periodic maintenance activities. Assist Engineer on machine improvement and parts evaluation. Setup equipment maintenance SOP and its work procedure. Repair and troubleshoot on machine breakdown. Manage spare parts inventory. Assist in any other related duties assigned by immediate supervisor as when necessary. Total Work Experience: 5 years and 11 months.']

    # Test queries
    queries = [
        "Job Title: Docker Kub Country: American Samoa Job Type: Permanent Qualification: Bachelorssss Languages requirement job: Chinese Job Description: seeking highly motivated skilled Docker Engineer manage optimize containerized applications infrastructure. ideal candidate deep experience Docker, container orchestration tools like Kubernetes, CI/CD pipelines. collaborate closely development operations teams ensure efficient containerization, deployment, scaling applications. Key Responsibilities: Containerization: Develop maintain Docker containers various applications microservices. Docker Management: Manage Docker images, containers, registries, ensuring scalability security. Orchestration: Implement manage container orchestration tools Kubernetes, Docker Swarm, OpenShift. CI/CD Pipelines: Integrate Docker containers CI/CD pipelines using tools like Jenkins, GitLab CI, Azure DevOps. Monitoring & Logging: Set monitoring logging solutions container environments (e.g., Prometheus, Grafana, ELK). Security: Implement security best practices containers, images, registries (e.g., vulnerability scanning). Performance Optimization: Analyze fine-tune Docker infrastructure optimal performance resource utilization. Collaboration: Work developers optimize applications containerization. Troubleshooting: Resolve container-related issues provide support production systems. Documentation: Create maintain technical documentation related container workflows standards. Skills & Qualifications: Strong experience Docker containerization applications. Hands-on experience container orchestration tools like Kubernetes, Docker Swarm, OpenShift. Expertise CI/CD tools Jenkins, GitLab CI, Azure DevOps. Knowledge Linux operating systems shell scripting. Experience cloud platforms (AWS, Azure, GCP) deploying containers cloud environments. Strong understanding networking concepts related containers (e.g., overlay networks). Familiarity security practices containers registries (e.g., image scanning tools like Trivy). Knowledge monitoring logging tools like Prometheus, Grafana, ELK Stack. Ability troubleshoot complex production issues related container environments. Excellent problem-solving skills ability work fast-paced environment. Education & Experience: Bachelor"
        "s degree Computer Science, Engineering, related field (or equivalent experience). 3+ years experience working Docker containerized applications. Certification Docker, Kubernetes, cloud platforms (e.g., CKA, Docker Certified Associate) plus. Preferred Qualifications: Experience Infrastructure Code (IaC) tools like Terraform Ansible. Familiarity microservices architecture best practices. Knowledge serverless technologies hybrid cloud deployments. Hands-on experience GitOps methodologies. Offer: Competitive salary benefits package. Opportunity work cutting-edge containerization technologies. Flexible work environment remote options. Access training professional development resources."
    ]

    # Tokenize corpus
    tokenized_corpus = [doc.lower().split() for doc in corpus]

    # Initialize BM25 with adjusted parameters
    bm25 = BM25Okapi(
        tokenized_corpus,
        # b= 0.75,
        # k1= 1.2
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

        print("\n" + "=" * 80)


def preprocess_text(text):
    # Lowercase, remove special characters, and split by whitespace
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()).split()


def split_into_segments(text, segment_size=50):
    # Split text into tokens, then group into segments of a certain size
    tokens = preprocess_text(text)
    return [tokens[i : i + segment_size] for i in range(0, len(tokens), segment_size)]


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
        scores.append(
            max(segment_score)
        )  # Take the highest score for each query segment

    # Calculate the average score across all segments
    average_score = sum(scores) / len(scores) if scores else 0
    return average_score





# def compare_documents_with_sentence_bm25(doc1, doc2):
#     # Split both documents into sentences
#     doc1_sentences = split_into_sentences(doc1)
#     doc2_sentences = split_into_sentences(doc2)

#     # Create a BM25 corpus with sentences from doc1
#     bm25 = BM25Okapi(doc1_sentences, b=0.75, k1=1.2)

#     # Score each sentence in doc2 against the sentences in doc1 corpus
#     scores = []
#     for query_sentence in doc2_sentences:
#         sentence_scores = bm25.get_scores(query_sentence)
#         scores.append(
#             max(sentence_scores)
#         )  # Take the highest score for each query sentence

#     # Calculate the average score across all sentences
#     average_score = sum(scores) / len(scores) if scores else 0
#     return average_score

from rank_bm25 import BM25Okapi

def split_into_sentences(text):
    # Split text into sentences and preprocess each sentence
    sentences = nltk.sent_tokenize(text)
    return [preprocess_text(sentence) for sentence in sentences]

def compare_documents_with_sentence_bm25(doc1, doc2, min_score=0, max_score=10):
    # Split both documents into sentences
    doc1_sentences = split_into_sentences(doc1)
    doc2_sentences = split_into_sentences(doc2)

    # Create a BM25 corpus with sentences from doc1
    bm25 = BM25Okapi(doc1_sentences, b=0.75, k1=1.2)

    # Score each sentence in doc2 against the sentences in doc1 corpus
    scores = []
    for query_sentence in doc2_sentences:
        sentence_scores = bm25.get_scores(query_sentence)
        scores.append(max(sentence_scores))  # Take the highest score for each query sentence

    # Calculate the average score across all sentences
    average_score = sum(scores) / len(scores) if scores else 0

    # Normalize scores using a fixed min-max range
    normalized_scores = [(score - min_score) / (max_score - min_score) if max_score > min_score else 0 for score in scores]

    # Calculate the normalized average score
    normalized_average_score = sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0

    return average_score, normalized_average_score



if __name__ == "__main__":
    # test_bm25_library()


    queries = [
        '''
        "Current Position1: Senior Cloud Engineer Current Location: Singapore Candidate Nationality: Singaporean Profile Summary: Seeking a challenging position involving technical and managerial skills in an environment where growth is possible. Primary Skill: AWS/Azure platforms, Infra as code using Terraform, Kubernetes, Container provisioning, OS level patching, Security concepts in hardening Primary Skills: AWS/Azure platforms, Infra as code using Terraform, Kubernetes, Container provisioning, OS level patching, Security concepts in hardening Skill 1: AWS/Azure platforms, Infra as code using Terraform, Kubernetes, Container provisioning, OS level patching, Security concepts in hardening Secondary Skills: Networking, TCP/IP, Routing Protocols, Documentation and Drawing of AWS Architecture, DevSecOps tooling and automation concepts, System programming & configuration, Technical application/architecture support, Active Directory, Database and script deployment, Architecture improvements, Security scanning and OS hardening, Proof of concept (POC) implementation, Administration level actions on Development, UAT, Staging and Production environments Candidate Education Details: Achieved BTech (Electronics Engineering) in Electrical Engineering from National University of Singapore (NUS), studied from 2021-01-01 to 2024-03-01. Achieved Diploma in Electronics in Electronics from Temasek Polytechnic, studied from 2014-01-01 to 2017-12-01. Achieved O Level from Changkat Changi Secondary School, studied from 2010-01-01 to 2013-12-01. Candidate Work Experience Details: Worked as Senior Cloud Engineer at Accenture Pte Ltd for 1 years 9 months. Role involved: Experience in architecting, designing, developing, and implementing cloud solutions on AWS/Azure platforms. Familiar with maintaining at scale enterprise online web architectures, microservices architectures. Able to design/build/run IaC tooling like terraform to automate environment build. Strong Experience and Working Knowledge working in AWS Cloud Environment such as VPC, Direct Connect, IAM, S3, EC2, EKS, Tagging, Cloud Monitoring, Config, CloudTrail, Lambda, ECS, CDN, Route53, Storage Gateway etc. Experience in Infra as code using Terraform, cloud formation, ansible and Shell scripts. Experience in Kubernetes and the various open-source security and monitoring technologies for Kubernetes e.g., Istio, Prometheus, Thanos etc. Experience in operation monitoring and has worked on ElasticSearch, Grafana, Fluentd. Experience in AWS Organization and Enterprise support. Experience in cost optimization with Reserved Instances, spot instances, cost explorer. Familiar with Networking, TCP/IP, Routing Protocols, Fiber Leased Lines Connectivity. Familiar with Linux/Windows based OS Provisioning, Maintenance and Support Operations. Familiar in Documentation and Drawing of AWS Architecture, Standards, and Operating Procedures. Consolidate account into AWS organisation. Define policies and standard for AWS organization. Familiar with the VPC architecture. Design transformation of on-prem connectivity to public cloud. Design and implement solution for multi-cloud interconnection architecture. Actively control and optimize AWS Cloud costs. Ensure Cloud environments are compliant to security policies. Perform scheduled maintenance activities for all related resources in AWS Cloud. Incident management and support during incidents and outages. Troubleshoot system and applications issues. Assess and Review AWS Cloud and Hybrid Environment for Optimisation and best practices. Perform SOP, Processes, Strategy and for AWS Cloud Environment. Management and monitoring of AWS Cloud Environment using relevant tools. Documentation and periodic reports on all AWS Cloud management, monitoring and maintenance activities. Document Cloud Inventory and Management. Demonstrating a strong desire and capability to learn, grow, and develop new skills for hybrid cloud. Familiar with container provisioning, images, repository lifecycle concepts & deployment of container centric services. Familiar with maintaining at enterprise Architectures including web application, Microservices. Familiar with DevSecOps tooling and automation concepts. Familiar with setting and defining operational processes. Familiar with OS level patching, Application-level patching, Operational housekeeping and archival, Backup and recovery etc. Familiar with gov level security concepts in hardening, scanning, security posture and mindset.. Familiar with Infra as Code using Ansible/Terraform Scripting. Worked as Infrastructure Engineer at NCS Pte Ltd for 8 months. Role involved: Plan, design, install, test & implement systems in accordance with specifications & service level. Where relevant, perform the necessary system programming & configuration. Perform and manage routine preventive maintenance and operational activities such as service requests, report generation, incident response, patch management. Perform Cloud infrastructure monitoring and escalation as per standard operations procedures. Assist the customer in solving issues related to Private Managed Cloud platforms.. Manage systems changes through established change request process & provide status reports to the relevant parties. Responsible for the system design, implementation, testing, managing and maintenance of proposed system. Support to users on a variety of system technologies including Microsoft Windows Server OS, Unix/Linux OS and CentOS. Provide technical application/architecture support to infra and application teams. Maintain and enhance application architecture components. Develop and maintain operation processes for Infrastructure deployment and maintenance. Generate regular report on the activities, performance to highlight the exceptions and fixes. Provide technical documentation support to Project & Solutions Teams, as well as hand over as-built system documentation to Maintenance support team. Manage and ensure timely patching deliverables conform to client patch management processes within the SLA given; define and provide test methodologies to test the patches; identify and manage risk and mitigation strategies; identify, analyses and ensure resolution of issues; manage change control. Scheduling, planning, resource allocation and prioritizing maintenance activities. Carry out monthly maintenance to back-up data and patch the Servers OS, monitor client performance measurement for client environments after every patching; and maintain records of the performance measurement taken. Provide 2nd line of support to address client related queries, incidents and requests reported on ground. Track system defects reported by users. Ensure prompt resolution is provided by the Contractors to the defects reported. Administer user, computer accounts, and group policies in the Active Directory to support technology exploration projects.. Work closely with Team Leader to meet system delivery milestones. Engage with back-line support and/or with vendor’s support to diagnose and rectify technical problems. Assist in day-to-day operation of the System Delivery team. Carry out other technical related duties that may be required. Support large scale IT infra environment. Maintain systems availability and performance. Worked as System Engineer at Xtremax Pte Ltd for 2 years. Role involved: Responsible for the setup, configuration, efficiency, hardening, deploying, automating, maintaining, managing and reliable operation of cloud, working AWS, Azure, Google Cloud. Ensure the uptime, performance, availability, scalability and security of the servers. Analyse, isolate and resolve both network and physical issues timely; make recommendations for future upgrades. Develop technical documentation and SOP procedures. Acquire, install, or upgrade computer components and software. Maintain backup including system backup, application backup, etc. Build, release, and configuration management of servers. Windows and Linux System Administration. Windows and Linux patch management. (Setup WSUS and Linux repo and push updates to clients). Database and script deployment for client sites. Pre-production Acceptance Testing to help assure the quality of our products/services. System troubleshooting and problem solving across platform and application domains. Suggesting architecture improvements, recommending process improvements. Security scanning and OS hardening. Ensuring critical system security through, the use of best-in-class cloud security solutions. Work with application and architecture teams to conduct proof of concept (POC) and implement the design in production environment. Fix deployment failures and find out root cause for the issues. Perform all kind of administration level actions on Development, UAT, Staging and Production environments. Worked as Associate Engineer at AFPD Pte Ltd for 1 years 6 months. Role involved: Perform equipment periodic maintenance activities. Assist Engineer on machine improvement and parts evaluation. Setup equipment maintenance SOP and its work procedure. Repair and troubleshoot on machine breakdown. Manage spare parts inventory. Assist in any other related duties assigned by immediate supervisor as when necessary. Total Work Experience: 5 years and 11 months."
    '''
    ]
    # queries =["Jane Doe. Dedicated and enthusiastic teacher with over 5 years of experience in fostering a positive and inclusive classroom environment. Skilled in curriculum development, lesson planning, and implementing innovative teaching techniques to accommodate diverse student needs. Passionate about helping students reach their full potential by promoting active learning and critical thinking. Education: Master of Education (M.Ed.) in Curriculum and Instruction, University of Education, City, State, Graduated: May 2020; Bachelor of Arts in English Education, State College, City, State, Graduated: May 2018. Professional Experience: English Teacher, City High School, City, State, August 2020 – Present - Teach English Language Arts to grades 9-12, designing engaging lessons that meet state standards and enhance critical thinking and literacy skills. Develop and adapt curriculum to accommodate diverse learning styles, creating a supportive and inclusive classroom environment. Assess student progress through formative and summative evaluations, providing regular feedback to students and parents. Incorporate technology and digital resources to foster interactive learning, resulting in a 20% increase in student engagement and participation. Serve as the advisor for the school’s journalism club, guiding students in publishing the school newspaper and developing their writing skills. Substitute Teacher, District School System, City, State, September 2018 – July 2020 - Covered various subjects across K-12, effectively managing classrooms and delivering prepared lessons. Adapted quickly to different classroom settings, ensuring a positive and productive learning environment for students of varying backgrounds and academic levels. Maintained clear communication with teachers and administrators, reporting on classroom activities and student progress. Skills: Curriculum Design & Lesson Planning - Experienced in creating interactive lessons that align with Common Core standards and encourage student participation; Classroom Management - Skilled at maintaining a structured and supportive classroom environment"]
    # queries =["Highly skilled DevOps Engineer with over 6 years of experience in implementing CI/CD pipelines, managing cloud infrastructure, and optimizing application deployment with Docker and Kubernetes. Proven ability to improve system reliability and streamline software delivery processes. Adept at collaborating with cross-functional teams to deliver high-quality solutions in fast-paced environments. Education: Bachelor of Science in Computer Science, Tech University, City, State, Graduated: May 2015. Professional Experience: DevOps Engineer, Tech Solutions Inc., City, State, June 2018 – Present - Designed and implemented automated CI/CD pipelines using Jenkins, reducing deployment time by 30%. Developed and managed containerized applications using Docker, orchestrated with Kubernetes for efficient scaling and resource utilization. Configured and monitored AWS infrastructure, including EC2, RDS, and S3, optimizing costs by 15%. Collaborated closely with developers to troubleshoot issues and optimize application performance. Created robust monitoring and logging solutions with Prometheus and Grafana to enhance system reliability. DevOps Specialist, Cloud Innovators LLC, City, State, September 2015 – May 2018 - Built and maintained automated build and deployment processes, ensuring rapid and reliable releases. Utilized Docker to containerize applications, creating standardized environments for development and testing. Set up continuous integration workflows with GitLab CI, enhancing code quality and reducing bugs. Automated server provisioning and configuration management with Ansible, saving time on manual processes. Skills: CI/CD Pipelines - Experienced with Jenkins, GitLab CI, and GitHub Actions for end-to-end automation; Docker & Kubernetes - Expert in containerization and orchestration; Cloud Management - Proficient in AWS, Azure; Configuration Management - Skilled in Ansible, Terraform for infrastructure as code; Monitoring & Logging - Knowledgeable in Prometheus, Grafana, and ELK Stack for system health tracking"]
    # queries = ['Experienced financial analyst with a strong background in accounting, data analysis, and reporting. Proficient in financial modeling, budgeting, and forecasting. Skilled in tools like Excel, SAP, and Power BI to analyze and visualize financial data. Demonstrated ability to work in high-pressure environments, providing insights to guide business decisions. Excellent communication skills and experience collaborating with stakeholders to deliver financial reports and strategy recommendations.']
    # queries=['''
    #       Candidate Basic Information2:
    # Current Position: Infrastructure Technical Architect
    # Current Location: Wiltshire, UK
    # Primary Skill: TOGAF,AWS,Azure,VMware,DevOps,Python,Docker,JSP440,MECM,REST_API,NIST,SCCM
    # Skill 1: TOGAF,AWS,Azure,VMware,DevOps,Python,Docker,JSP440,MECM,REST_API,NIST,SCCM
    # Secondary Skills: Questor,Jira,Confluence,AGILE,React,Kubernetes,MySQL,Elastic Search,Ansible,Nagios,IaaS,PaaS,IAM,SAML,Win server,Snowflake,Node.js,Open Shift,Zackman,VDI,OCI,Django,GOV,OpenID,KONGAPI,KONGOPS,KAFTA,HASHICORP,DYN365,DATALAKE,DATAWAREHOUSE,NODEJS,NSX,365,SQL,RHEL 7,CISCO,ASA,ITIL,NIST,GIAC Python Coder,ISO 27001 Foundation,ITIL Service Operation,CISSP,AWS Certified Machine Learning
    # Candidate Education Details:

    # Candidate Work Experience Details:
    # Worked as Lead Technical/Data Architect at Vanguard Investments for 6 months.
    # Role involved: Apply architecture and harmonize an infrastructure network with AWS stretched WANs. Correct and analyze Data and build a team for processing to the business strategy. Analyze MySQL Databases into Dynamics 365 and consolidate data into Cloud or VMware environment. Create a hardening security policy to align with standard company policies and compliance. Build POC and UAT test framework, resolve issues within the team. Provide formal guidance to cover stakeholder alignment on key areas.

    # Worked as Technical Architect at RAF Northolt for 2 months.
    # Role involved: Lead system architecture of Hardware and Software for MS Windows Server. Manage Active Directory Services, DNS, and DHCP servers Architecture and documentation within VMware technologies. Maintain close liaison with other technicians and engineers. Design and apply systems leveraging automation technologies, understanding of Virtualization and Container technologies using Python or Ansible. Assist design team in complex systems development.

    # Worked as Technical Architect MOD contract at Sopra Steria for 3 months.
    # Role involved: Lead Migration of CASS apps from JPA to Oracle OCI cloud. Migration of existing VMware environments. Risk assessment reporting/creation. Enhanced security management on domain/cyber awareness. Focus on compliance with JSP440/NIST standards.

    # Worked as Technical Security Architect at Defence Academy - Serco for 3 months.
    # Role involved: Provide technical solutions and security rules for implementing business objectives. Replace current configuration management processes. Evaluate and implement new systems to replace out of scope systems. Provide detailed guidelines on MECM implementation and Azure integration. Formulate architecture documents for stakeholder signoff.

    # Worked as Technical Architect at QinetiQ for 1 months.
    # Role involved: Design and define VMware ESXi 7 on new remote system for customer. Produce HLD and PTM/LLD documents for VSAN deployment. Meeting compliance and MOD security policy on deployment to the RAF test firing range. Amend security processes and Forefront security migration.

    # Worked as Technical Architect at RAF Northolt.
    # Role involved: Design and install secure LAMP server for in-house systems. Migrate to current and previous versions to run on VMware stack. Define update server with access to perimeter network. Plan AWS migration solutions from AZURE on private cloud.

    # Worked as Technical Security Architect at UAE/Saudi Blakwatch Ltd for 9 months.
    # Role involved: Enhance security of command & control systems with Python for logistical software. Lead Data Architecture analytical governance. Design AD/VMware AD and POC Cloud DNS. Collaborate on IT security, compliance processes. Lead stakeholder engineering concepts for IT security and governance.

    # Worked as Infrastructure Technical Architect at C3IA/BAE for 1 years 5 months.
    # Role involved: Work on Ministry of Defence project. Implement Django automation technologies on a development platform. Design and implement Cloud integration concepts on secure data. Lead configuration management and capacity management. Lead penetration testing concepts within security framework.

    # Worked as Infrastructure Security Analyst/Architect at KPMG Pty Ltd Hong Kong for 4 months.
    # Role involved: Design secure networks and perimeter load balanced architecture. Document repository LLD design and POC integration. Manage VMware virtualization and networking perimeters. Lead stakeholder management for adopted security practices.

    # Worked as Security Infrastructure Solutions at Airbus - Defence & Space for 11 years 5 months.
    # Role involved: Implement Hardware load balancing solution on secure systems. Lead Solution Platform migration for Geo-spatial platforms. Lead SCOM/SCCM implementation into Ministry Of Justice systems. Provide ITIL compliance documentation on projects. Lead network security enhancements on platform applications.

    # Total Work Experience: 15 years and 2 months.

    #     ''']

#     queries = ['''
# Candidate Basic Information:
# Current Position: Software Developer
# Current Location: Singapore
# Candidate Nationality: Singaporean
# Profile Summary: An aspiring software developer eager to collaborate and develop tailored products leveraging full-stack skills acquired through Le Wagon Web Development bootcamp and Avensys Consulting. Unique background mirroring the multi-faceted demands of software development - a Mechanical Aerospace Engineer turned software developer enthusiast. Graduate in NUS’ Master in Mechanical engineering and over seven years of experience as a test engineer with Defense Science Organization National Laboratories (DSO), encompassing a strong background in MATLAB, LABView and EXCEL VBA to design, run test projects and engineering computation for data analysis. Was also a recipient of DSO group performance award for three consecutive years (2020 to 2022).
# Primary Skill: Java,Spring,React,Matlab,LabView,Excel VBA
# Skill 1: Java,Spring,React,Matlab,LabView,Excel VBA
# Secondary Skills: MS Office,SolidWorks,ProEngineering,ANSYS,AutoCAD,Dewesoft,HTML,CSS,SASS,JavaScript,Typescript,Ruby,Python,OOP,REST API,SQL,PostgreSQL,MongoDB,firebase firestore and storage,cloudinary,Ruby on Rails,NodeJS,Stimulus,Angular,Figma,Adobe Photoshop,Adobe Lightroom,Git,Github,Photography,Drawing,Photo editing
# Languages: English
# Candidate Education Details:
# Achieved M.S.c in Mechanical Engineering in Mechanical Engineering (Oil and Gas) from National University of Singapore (NUS), studied from 2015-08 to 2016-05.

# Achieved BEng (Hons) of Mechanical Engineering in Mechanical Engineering (Aerospace) from National University of Singapore (NUS), studied from 2012-08 to 2015-05.

# Achieved Diploma in Mechanical Engineering with merit in Mechanical Engineering from Singapore Polytechnic (SP), studied from 2007-04 to 2010-04.

# Achieved Diploma Plus in Higher Mathematics in Higher Mathematics from Singapore Polytechnic (SP), studied from 2007-04 to 2010-04.

# Candidate Work Experience Details:
# Worked as Software Developer at Avensys Consulting Pte Ltd.
# Role involved: Completed 8 weeks of Java full stack web development training in core Java and spring framework for backend and frontend web technologies framework such as React and Angular.. Developed a simple employee management portal that allows users to create an account, login and manage all employees in the app database and perform CRUD functions.. Developed a social media web full stack web application using spring boot, spring security (JWT auth) and MySQL database for backend and angular as the frontend framework.. Developed a CV parser full stack web application that aims to parse and extract specific fields from Word/PDF format CVs, including name, email, mobile number, skills, employment experiences, and recent 3 companies..

# Role involved: Developed a Meet-up web application clone in both frontend and backend (ruby on rails), which integrated third party packages (Pg search and mapbox).. Led a team to develop full stack web application (ruby on rails) called Hello World! It is a CRUD itinerary planner web application with PostgreSQL as the database..

# Worked as Senior Member of Technical Staff at DSO National Laboratories for 5 years 11 months.
# Role involved: Developed scripts using MATLAB for calibration processes and experimental tests to acquire accurate test results during wind tunnel testing.. Developed scripts using MATLAB to process and reduce large experimental raw data for various aerodynamic wind tunnel tests for analysis.. Designed mechanical iterative process using MATLAB as well as python, in conjunction with simulation results obtained from SOLIDWORKS finite element simulation.. Wrote EXECL VBA code with intuitive user interface to easily read in CSV data/text files and cross plot data accompanied with user selectable chart display for analysis.. Implemented pressure system hardware integration with LABView code for pressure system calibration and data acquisition.. Expanded testing capabilities for wind tunnel system by introducing new controls code logic and modifying existing code.. Advised multiple project teams on technical aspects of aerodynamics testing which contributed to the success of tests conducted in the wind tunnel facility.. Utilized and configured Dewesoft DAQ system for data acquisition and 1st level data processing for various aerodynamic wind tunnel testing.. Planned and coordinated with external vendors on facility’s hardware maintenance and upgrade, and software engineers for facility’s software system upgrade..

# Worked as HIMARS Officer at Singapore Armed Forces (SAF) for 2 years .
# Role involved: Led a platoon of 16 specialists and 2 men to achieve readiness condition 1 for field artillery evaluation (FATEP)..

# Total Work Experience: 8 years and 1 months.
#               ''']
    # Test queries
    corpus = [
       '''
Job Title: Docker Kub Country: American Samoa Job Type: Permanent Qualification: Bachelorssss Languages requirement job: Chinese Job Description:  │
│ │   seeking highly motivated skilled Docker Engineer manage optimize containerized applications infrastructure. ideal candidate deep experience Docker, container │
│ │   orchestration tools like Kubernetes, CI/CD pipelines. collaborate closely development operations teams ensure efficient containerization, deployment, scaling │
│ │   applications. Key Responsibilities: Containerization: Develop maintain Docker containers various applications microservices. Docker Management: Manage Docker │
│ │   images, containers, registries, ensuring scalability security. Orchestration: Implement manage container orchestration tools Kubernetes, Docker Swarm,        │
│ │   OpenShift. CI/CD Pipelines: Integrate Docker containers CI/CD pipelines using tools like Jenkins, GitLab CI, Azure DevOps. Monitoring & Logging: Set          │
│ │   monitoring logging solutions container environments (e.g., Prometheus, Grafana, ELK). Security: Implement security best practices containers, images,         │
│ │   registries (e.g., vulnerability scanning). Performance Optimization: Analyze fine-tune Docker infrastructure optimal performance resource utilization.        │
│ │   Collaboration: Work developers optimize applications containerization. Troubleshooting: Resolve container-related issues provide support production systems.  │
│ │   Documentation: Create maintain technical documentation related container workflows standards. Skills & Qualifications: Strong experience Docker               │
│ │   containerization applications. Hands-on experience container orchestration tools like Kubernetes, Docker Swarm, OpenShift. Expertise CI/CD tools Jenkins,     │
│ │   GitLab CI, Azure DevOps. Knowledge Linux operating systems shell scripting. Experience cloud platforms (AWS, Azure, GCP) deploying containers cloud           │
│ │   environments. Strong understanding networking concepts related containers (e.g., overlay networks). Familiarity security practices containers registries      │
│ │   (e.g., image scanning tools like Trivy). Knowledge monitoring logging tools like Prometheus, Grafana, ELK Stack. Ability troubleshoot complex production      │
│ │   issues related container environments. Excellent problem-solving skills ability work fast-paced environment. Education & Experience: Bachelor's degree        │
│ │   Computer Science, Engineering, related field (or equivalent experience). 3+ years experience working Docker containerized applications. Certification Docker, │
│ │   Kubernetes, cloud platforms (e.g., CKA, Docker Certified Associate) plus. Preferred Qualifications: Experience Infrastructure Code (IaC) tools like Terraform │
│ │   Ansible. Familiarity microservices architecture best practices. Knowledge serverless technologies hybrid cloud deployments. Hands-on experience GitOps        │
│ │   methodologies. Offer: Competitive salary benefits package. Opportunity work cutting-edge containerization technologies. Flexible work environment remote      │
│ │   options. Access training professional development resources. 
'''
    ]

    # queries = ['Experienced financial analyst with a strong background in accounting, data analysis, and reporting. Proficient in financial modeling, budgeting, and forecasting. Skilled in tools like Excel, SAP, and Power BI to analyze and visualize financial data. Demonstrated ability to work in high-pressure environments, providing insights to guide business decisions. Excellent communication skills and experience collaborating with stakeholders to deliver financial reports and strategy recommendations.']
    # queries = ['Hello world. What are you guys doing tonight? Any one up for some dinner? Hello world. What are you guys doing tonight? Any one up for some dinner? Hello world. What are you guys doing tonight? Any one up for some dinner? Hello world. What are you guys doing tonight? Any one up for some dinner? Hello world. What are you guys doing tonight? Any one up for some dinner? Hello world. What are you guys doing tonight? Experienced financial analyst with a strong background in accounting. ']

    score = compare_documents_with_sentence_bm25(corpus[0], queries[0])
    print(score)
    # print("Normalized Score: ", sigmoid(score, alpha=1))
