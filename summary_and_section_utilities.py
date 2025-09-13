from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Prompts Utilities")

class ResumeSections(BaseModel):
    Work_Experience: str = Field(..., alias="Work Experience", description="Full content of the Work Experience section")
    Education: str = Field(..., alias="Education", description="Full content of the Education section")
    Skills: str = Field(..., alias="Skills", description="Full content of the Skills section")
    Internship_and_Academic_Projects: str = Field(..., alias="Internship and Academic Projects", description="Full content of the Internship and Academic Projects section")
    Certifications_Awards_and_Recognitions: str = Field(..., alias="Certifications, Awards and Recognitions", description="Full content of the Certifications, Awards and Recognitions section")

    class Config:
        allow_population_by_field_name = True

def resume_summarization(resume_text, llm_model):
    template = """
    You are given a resume.
    Write a concise written professional summary in the following format:

    Line 1:
    Name | Current Designation | High level tech stack | Total years and months of experience


    Then provide 4–5 short bullet points highlighting the entire resume, covering:
    - Current and previous organizations (with focus on recent roles)
    - Key responsibilities or strengths
    - Major projects or domains worked in
    - Broad technical/functional expertise
    - Notable achievements


    Guidelines:
    - Keep bullets short (1 line each, no long paragraphs).
    - Do not list the full tech stack. Keep it high level summarizing everything.
    - Ensure output is clear, scannable, and recruiter-friendly.
    \n{resume}
    """

    prompt = PromptTemplate(input_variables=['resume'], template=template)
    output_parser = StrOutputParser()
    formatted_prompt = prompt.format(resume=resume_text)
    raw_output = llm_model.invoke(formatted_prompt)
    logger.info(raw_output)
    summarised_text = output_parser.invoke(raw_output)
    logger.info(summarised_text)

    return summarised_text

def get_prompt_and_json_parser_for_sections():
    from langchain_core.output_parsers import JsonOutputParser
    parser = JsonOutputParser(pydantic_object=ResumeSections)

    prompt_template = PromptTemplate(
        input_variables=["resume"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template="""
            You are a helpful resume parser. Given the resume below, extract and return the following sections:
            - Work Experience
            - Education
            - Skills
            - Internship and Academic Projects
            - Certifications, Awards and Recognitions

            Take extra precautions for below:
                - If a section is not found, output: "Section Not Present".
                - Include the entire section if present in the resume.
                - Do not infer or create content that is not explicitly stated in the resume.
                - Avoid '\n' character in key value pair.

            {format_instructions}

            Resume:
            {resume}
            """
            )
    return prompt_template, parser

def get_prompt_and_json_parser_for_comparison_table(is_jd):
    from langchain_core.output_parsers import JsonOutputParser
    parser = JsonOutputParser(pydantic_object=ResumeSections)

    if is_jd:
        prompt_template = PromptTemplate(
            input_variables=["jd"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            template="""
                You are a helpful Job Description Parser. Given the job descrition below, extract and return the following sections:
                - Work Experience (in years)
                    • Only provide overall work experience in years
                - Education (highest qualification only)
                - Skills
                    • Summarize into max 8-9 comma-separated items with the must needed skills.

                Take extra precautions for below:
                    - Do not infer or create content that is not explicitly stated in the resume.
                    - Work experience can also be written as experience

                {format_instructions}

                Job Description:
                {jd}
                """
                )
    else:
        prompt_template = PromptTemplate(
            input_variables=["resume", "jd"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            template="""
                You are a helpful resume parser. Given the resume and job description below, extract and return the following sections from resume. Refer Job Description just as reference for Skills:
                - Work Experience (in years and months)
                    • Only provide overall work experience in years and months and no textual content
                - Education (highest qualification only)
                - Skills:
                    • Divide into two parts - list only JD-required skills and If any required skills are missing, explicitly add 'Missing: <comma separated skills>'.
                - Internship and Academic Projects
                    • Only provide header and limit to 2–3 items if available).
                - Certifications, Awards and Recognitions
                    • Only provide certificate subject

                Take extra precautions for below:
                    - If a section is not found, output: "Section Not Present".
                    - Do not infer or create content that is not explicitly stated in the resume.
                    - Calculate Work experience based on tenors worked in companies

                {format_instructions}

                Resume:
                {resume}

                Job Description:
                {jd}
                """
                )
    return prompt_template, parser