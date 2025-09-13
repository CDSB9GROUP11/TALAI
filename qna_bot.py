from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

def chatprompt():
    chat_prompt_template = ChatPromptTemplate([
         ('system', 'You are good resume parser expert and helps in answering questions. Answer the question from the provided resume. Make sure to provide all the details, if the answer is not in the provided resume just reply back, "Answer is not available in provided resume", dont provide the wrong answers or dont hallucinate the answers.\n Resume: \n {resume}?'),
         ('human', 'Question: \n {question}\nAnswer:')
        ])
    return chat_prompt_template

def handle_user_input(user_question, llm_model, embedding_model):
    new_db = FAISS.load_local("/content/faiss_selected_resume", embeddings=embedding_model, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    resume_text = "\n\n".join([doc.page_content for doc in docs])
    print(resume_text)
    chain = chatprompt() | llm_model | StrOutputParser()
    response = chain.invoke({
        "resume": resume_text,
        "question": user_question
    })
    return response

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