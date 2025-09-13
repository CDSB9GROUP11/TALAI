from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
import os
import pandas as pd
import math
import logging

##User defined libraries and packages

from llm_utilities import *
from nosql_utilities import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ParserUtilities")

#####################################################################################################################
# Start of Utility functions for "Data Injection" page
#####################################################################################################################

def load_resume_file(file_path: str, file_id: str):
    """
    This function takes the file path & combines the resume in single page content.
    It updates the id and metadata for the resume(Document Object) & retruns the resume.
    Each resume is Document object.(class 'langchain_core.documents.base.Document')
    Document object has 4 attributes - metadata,page_content,id,type
    Sample file path name: InformationTechnology_<noticeperiod>_<gender>
    """
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    # If resume has a multiple pages then combine pages. This is handlled in below if statement
    if len(docs) > 1:
        #full_text containes combine text from each page in multipage resume
        full_text = "\n".join([d.page_content for d in docs])

        #assigning full_text to page[0] content
        docs[0].page_content = full_text

        #Changing metadata for page number. Setting page number as 'Combined'
        docs[0].metadata['page']='Combined'

    #Keeping one page only
    docs[0].metadata['title'] = file_id

    #Split file_id to get gender, notice_period(if available) and subject
    file_splits = file_id.split("_")
    if len(file_splits) == 4:
        notice_period, subject, gender = file_splits[1], file_splits[0], file_splits[2]
    else:
        notice_period, subject, gender = 'Not Found', file_splits[0], file_splits[1]

    docs[0].metadata['notice_period'] = notice_period
    docs[0].metadata['gender'] = gender
    docs[0].metadata['subject'] = subject #Unique resume identifier
    docs[0].id = file_id #Update the id
    return docs[0]

def get_section_headings():
    """
    This function returns the list of section headings.
    Resumes will be divided into the sections present in this list
    """
    section_list = ['Work Experience', 'Education', 'Skills', 'Internship and Academic Projects', 'Certifications, Awards and Recognitions']
    return section_list

def create_vector_storing_dataframes():
    """
    This function creates the dataframe for storing the vector embeddings.
    """
    import pandas as pd
    column_names = ["id", "vector", "metadata", "section_text"]

    # Create an empty DataFrame with these column names
    df_resume = pd.DataFrame(columns=column_names)
    df_sections = pd.DataFrame(columns=column_names)
    return df_resume, df_sections

def update_section_vectors_in_df(embedding_llm, json_prompt_output, section_list, df_sections, file_id, session):
    """
    This function invokes the embedding model on each section of resume.
    It updates the section dataframe with vector values.
    It returns the updated dataframe.
    """
    for section in section_list:
        section_row_id = file_id+"_"+section
        logger.info(f"Generating embedding for {section_row_id}")
        # section_text = json_prompt_output["sections"][section]
        section_text = json_prompt_output[section]
        section_vector = embedding_llm.embed_query(section_text)
        section_metadata = {'metadata_id':section_row_id,
                            'resume_id': file_id}
        section_row = pd.DataFrame(
            {'id': [section_row_id],
             'vector': [section_vector],
             'metadata':[section_metadata],
             'section_text':[section_text]
             })
        insert_resume_data(session, section_row)
        df_sections = pd.concat([df_sections, section_row], ignore_index=True)
    return df_sections

def dataframe_to_documents(df: pd.DataFrame) -> list:
    """
    Converts a DataFrame with 'section_text' and 'metadata' columns into LangChain Documents.
    """
    documents = []
    for _, row in df.iterrows():
        doc = Document(
            page_content=row["section_text"],
            metadata=row["metadata"]
        )
        documents.append(doc)
    return documents

def build_vector_stores(df_sections: pd.DataFrame, df_resume: pd.DataFrame, embedding_model):
    """
    Converts both section-level and resume-level DataFrames into FAISS stores and saves them.
    """
    # Convert to LangChain documents
    section_docs = dataframe_to_documents(df_sections)
    resume_docs = dataframe_to_documents(df_resume)

    # Create and save FAISS stores
    section_store = create_and_save_faiss_store_from_documents(section_docs, embedding_model, "faiss_sections_store")
    resume_store = create_and_save_faiss_store_from_documents(resume_docs, embedding_model, "faiss_resume_store")

    return section_store, resume_store

def update_resume_vector_in_df(embedding_llm, resume, df_resume, session):
    """
    This function invokes the embedding model on entire resume.
    It updates the resume dataframe with vector values
    """
    logger.info(f"Generating embedding for {resume.id}")
    resume_vector = embedding_llm.embed_query(resume.page_content)
    resume_row = pd.DataFrame(
            {'id': [resume.id],
             'vector': [resume_vector],
             'metadata':[resume.metadata],
             'section_text':[resume.page_content]
             })
    insert_resume_data(session, resume_row)
    df_resume = pd.concat([df_resume, resume_row], ignore_index=True)
    return df_resume

def convert_to_ranked_df(df):
    """
    Converts a DataFrame with candidate IDs and scores into a ranked format.
    Columns: Rank, Resume, Score
    """
    df_ranked = pd.DataFrame({
        "Rank": range(1, len(df) + 1),
        "Resume": df["id"],
        "Score": df["weighted_sum_of_similarity_scores"]
    }).reset_index(drop=True)
    return df_ranked

#####################################################################################################################
# End of Utility functions for "Data Injection" page
#####################################################################################################################

#####################################################################################################################
# Start of Utility functions for "Candidate Search and Screening Tool" page
#####################################################################################################################

def is_sum_of_all_weighing_factors_one (weights):
    """
    This function checks if sum of all weighing factors is equal to one.
    It returns boolean output
    """
    sum = 0.0
    for value in weights.values():
        sum = sum + value
    return math.isclose(sum, 1.0, rel_tol=1e-5, abs_tol=1e-8)


def get_filter_condition_for_metadata_filtering(job_area):
    """
    This function filters the candidate profiles to avoid calculating similarity score with each record in the pinecone index.
    This filter condition applies on the metadata of the record, hence called metadata filtering
    """
    filter_condition = {"subject": job_area}
    return filter_condition

def create_similarity_score_dataframes():
    """
    This function creates two dataframes for storing the similarity scores between
    1. job description & each section of resume
    2. job description & entire resume
    """
    import pandas as pd
    column_names = ["id", "section_title", "similarity_score"]

    # Create empty DataFrames with these column names
    df_sectionwise_similarity_score = pd.DataFrame(columns=column_names)
    df_resume_similarity_score = pd.DataFrame(columns=column_names)
    return df_resume_similarity_score,df_sectionwise_similarity_score

def coarse_filter_resumes_faiss(
    faiss_store,
    query_embedding: list[float],
    top_k: int,
    df_resume_similarity_score: pd.DataFrame
) -> pd.DataFrame:
    """
    Filters candidate profiles using the FAISS resume store.
    Computes similarity scores between job description and resume embeddings,
    and updates the similarity_score_dataframe.

    Args:
        faiss_store (FAISS): FAISS vector store containing resume-level embeddings.
        query_embedding (list[float]): Embedding vector of the job description.
        top_k (int): Number of top matches to retrieve.
        df_resume_similarity_score (pd.DataFrame): DataFrame to append similarity scores.

    Returns:
        pd.DataFrame: Updated similarity score DataFrame.
    """
    # Perform similarity search
    results = faiss_store.similarity_search_with_score_by_vector(query_embedding, k=top_k)
    for match, score in results:
        metadata = match.metadata or {}
        section_title = metadata.get("title", "resume")
        doc_id = metadata.get("title", "unknown")

        similarity_score_row = pd.DataFrame({
            'id': [doc_id],
            'section_title': [section_title],
            'similarity_score': [score]
        })

        df_resume_similarity_score = pd.concat(
            [df_resume_similarity_score, similarity_score_row],
            ignore_index=True
        )

    return df_resume_similarity_score

def compute_sectionwise_similarity_scores_faiss(
    faiss_store,
    query_embedding: list[float],
    resume_id_list: list[str],
    df_sectionwise_similarity_score: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculates cosine similarity scores between the query embedding and each section
    in the FAISS store for the given resume_id_list. Filters results manually based on metadata.

    Args:
        faiss_store (FAISS): FAISS vector store containing section-level embeddings.
        query_embedding (list[float]): Embedding vector of the job description.
        resume_id_list (list[str]): List of resume IDs to filter sections.
        df_sectionwise_similarity_score (pd.DataFrame): DataFrame to append similarity scores.
        top_k (int): Number of top matches to retrieve from FAISS.

    Returns:
        pd.DataFrame: Updated similarity score DataFrame.
    """
    # Retrieve top_k matches from FAISS
    # results = faiss_store.similarity_search_by_vector(query_embedding, k=top_k)
    results = faiss_store.similarity_search_with_score_by_vector(query_embedding, k=10000)

    for match, score in results:
        metadata = match.metadata or {}
        resume_id = metadata.get("resume_id")
        section_id = metadata.get("metadata_id", "unknown")

        # Manual metadata filtering
        if resume_id in resume_id_list:
            similarity_score_row = pd.DataFrame({
                'id': [resume_id],
                'section_title': [section_id],
                'similarity_score': [score]
            })

            df_sectionwise_similarity_score = pd.concat(
                [df_sectionwise_similarity_score, similarity_score_row],
                ignore_index=True
            )

    return df_sectionwise_similarity_score

def calculate_weighted_combined_similarity_score(distinct_candidates, df_combined, weightage_by_section, df_candidate_weighted_score):
    """
    This function calculates weighted sum of similarity scores for each candidate.
    """
    for c in distinct_candidates:
        weighted_score = df_combined.loc[df_combined["section_title"] == c, "similarity_score"].iloc[0] * weightage_by_section["Entire Resume"]/100
        + df_combined.loc[df_combined["section_title"] == c+"_Work Experience", "similarity_score"].iloc[0] * weightage_by_section["Work Experience"]/100
        + df_combined.loc[df_combined["section_title"] == c+"_Education", "similarity_score"].iloc[0] * weightage_by_section["Education"]/100
        + df_combined.loc[df_combined["section_title"] == c+"_Skills", "similarity_score"].iloc[0] * weightage_by_section["Skills"]/100
        + df_combined.loc[df_combined["section_title"] == c+"_Internship and Academic Projects", "similarity_score"].iloc[0] * weightage_by_section["Internship and Academic Projects"]/100
        + df_combined.loc[df_combined["section_title"] == c+"_Certifications, Awards and Recognitions", "similarity_score"].iloc[0] * weightage_by_section["Certifications, Awards and Recognitions"]/100

        weighted_score_row = pd.DataFrame(
            {'id': [c],
             'weighted_sum_of_similarity_scores': [weighted_score]
             })

        df_candidate_weighted_score = pd.concat([df_candidate_weighted_score, weighted_score_row], ignore_index=True)
    return df_candidate_weighted_score

#####################################################################################################################
# End of Utility functions for "Candidate Search and Screening Tool" page
#####################################################################################################################