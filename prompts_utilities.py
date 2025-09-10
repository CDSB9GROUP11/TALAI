
import sys
sys.path.append('/content/drive/MyDrive/CDS Collab Notebooks/Capstone/CodePipelines3/')

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
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

def get_conversation_chain(llm_model):
    chat_prompt_template = ChatPromptTemplate([
         ('system', 'You are good resume parser expert and helps in answering questions. Answer the question from the provided resume. Make sure to provide all the details, if the answer is not in the provided resume just reply back, "Answer is not available in provided resume", dont provide the wrong answers or dont hallucinate the answers.\n Resume: \n {resume}?'),
         ('human', 'Question: \n {question}\nAnswer:')
        ])
    chain = chat_prompt_template|llm_model|StrOutputParser()
    return chain

def resume_summarization(resume_text, llm_model):
    template = '''Generate a Resume Summary Script in 5-6 bullet points for an engaging video presentation aimed at a hiring manager.
    - Tone: Friendly, Confident, Assertive.
    - Base the script strictly on the given resume content — do not hallucinate or add unverifiable details.\n{resume}'''

    prompt = PromptTemplate(input_variables=['resume'], template=template)
    output_parser = StrOutputParser()
    formatted_prompt = prompt.format(resume=resume_text)
    raw_output = llm_model.invoke(formatted_prompt)
    logger.info(raw_output)
    summarised_text = output_parser.invoke(raw_output)
    logger.info(summarised_text)

    return summarised_text


def notice_period_prompt(llm_model):
    chat_prompt_template = ChatPromptTemplate([
         ('system', 'You are good and helpful assistant and helps in answering questions. Find out the notice period from public available sources for most recent organization provided in resume. Make sure to provide to provide accurate details and provide the source link at the end of answer, dont provide the wrong answers or dont hallucinate the answers. \n Resume: \n {resume}?'),
         ('human', 'Question: \n Notice Period of the most recent job organization the candidate working for. \nAnswer:')
        ])
    chain = chat_prompt_template|llm_model|StrOutputParser()
    return chain

def handle_notice_period(embedding_model, llm_model):
    user_question = "Notice Period of the most recent job organization the candidate working for"
    new_db = FAISS.load_local("/content/faiss_selected_resume",  embeddings=embedding_model, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = notice_period_prompt(llm_model)
    response = chain.invoke({
        "resume": docs
    })
    return response

# def handle_user_input(user_question, llm_model, embedding_model):
#     new_db = FAISS.load_local("/content/faiss_selected_resume", embeddings=embedding_model, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversation_chain(llm_model)
#     response = chain.invoke({
#         "resume": docs,
#         "question": user_question
#     })
#     return response

def handle_user_input(user_question, llm_model, embedding_model):
    new_db = FAISS.load_local("/content/faiss_selected_resume", embeddings=embedding_model, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Extract text content from each Document
    # resume_text = "\n\n".join([doc.page_content for doc in docs])
    resume_text = "\n\n".join([f"{doc.metadata.get('source', '')}:\n{doc.page_content}" for doc in docs])


    chain = get_conversation_chain(llm_model)
    response = chain.invoke({
        "resume": resume_text,
        "question": user_question
    })
    return response

def get_prompt_output_format() -> dict:
    """
    This function return the string containing the output format.
    So that everytime we will get result in same format.
    This will help to trace and process the output without incurring runtime errors.
    """
    output_format = {
    "sections": {
    "Work Experience": "<to be fetched>",
    "Education": "<to be fetched>",
    "Skills": "<to be fetched>",
    "Internship and Academic Projects": "<to be fetched>",
    "Certifications, Awards and Recognitions": "<to be fetched>"
    }
    }    # Need to define output format here
    return output_format

def get_prompt_template_to_divide_resumes_into_sections():
    """
    This function returns the prompt template to divide resumes into sections.
    """
    # Define a template for the prompt
    template = '''
    You are good and helpful resume parser and processor expert. You are given the resume \n "{resume}"
    Perform the following steps:

    1. Split the resume into the following sections:
        a. Work Experience.
        b. Education
        c. Skills
        d. Internship and Academic Projects
        e. Certifications, Awards and Recognitions
        - If a section is not found, output: "Section Not Present".
        - Include the entire section if present in the resume.
        - Do not infer or create content that is not explicitly stated in the resume.
    2. Return the output in "{output_format}" JSON format. Avoid '\n' character in key value pair.
    '''
    # Create a PromptTemplate object from the template
    prompt_template = PromptTemplate.from_template(template=template)
    return prompt_template

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
