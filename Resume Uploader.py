import streamlit as st
import pandas as pd
from io import BytesIO
import tempfile
import zipfile
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import time
import base64
from PIL import Image

### User defined packages/modules

load_dotenv()

from summary_and_section_utilities import *
from nosql_utilities import *
from pii_masker import redact_personal_information
from resume_uploader_utilities import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Resume Uploader")

# --- Extract pdfs from ZIP and save to temp dir ---
def extract_and_mask_from_zip(zip_bytes):
    temp_dir = tempfile.mkdtemp(prefix="resume_zip_")
    extracted_paths = []

    with zipfile.ZipFile(zip_bytes) as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.lower().endswith(".pdf"):
                with zip_ref.open(file_info) as source_file:
                    original_pdf = BytesIO(source_file.read())
                    original_pdf.name = os.path.basename(file_info.filename)

                redacted_pdf = redact_personal_information(original_pdf)

                # Save redacted PDF to temp
                temp_path = os.path.join(temp_dir, original_pdf.name)
                redacted_pdf.seek(0)
                with open(temp_path, "wb") as f:
                    f.write(redacted_pdf.read())

                extracted_paths.append(Path(temp_path))

    return extracted_paths

def safe_concat_with_schema(db_df, new_df):
    """
    Ensures both DataFrames match expected schema before vertical concat.
    Renames columns if needed and fills missing ones with NaN.
    """
    def align_schema(df):
        df = df.rename(columns={
            'resume_id': 'id',
            'embedding_vectors': 'vector'
        })
        return df

    df1_aligned = align_schema(db_df)

    return pd.concat([df1_aligned, new_df], ignore_index=True)


st.set_page_config("Upload Resumes | TalAI", layout="wide")

#Initializing session variables

if "database_session" not in st.session_state:
    st.session_state.database_session = setup_astra_session()
    create_resume_table(st.session_state.database_session)

if "section_list" not in st.session_state:
    st.session_state.section_list = get_section_headings()
if "embedding_llm" not in st.session_state:
    st.session_state.embedding_llm = mistralai_embedding_model()
if "llm_model" not in st.session_state:
    st.session_state.llm_model = mistralai_llm_model()
if "openai_embedding_llm" not in st.session_state:
    st.session_state.openai_embedding_llm = embedding_model()
if "openai_llm_model" not in st.session_state:
    st.session_state.openai_llm_model = llm_model()

# Track candidate searching and matching
if "matched_resume_count" not in st.session_state:
    st.session_state.matched_resume_count = 0
if "new_question_flag" not in st.session_state:
    st.session_state.new_question_flag = set()
if "processed_resumes" not in st.session_state:
    st.session_state.processed_resumes = set()


# ------------------ Session State Init ------------------

for key in ["selected_index", "selected_resume", "resume_summary_content", "comparison_table", "job_description", ""]:
    if key not in st.session_state:
        st.session_state[key] = None

state_keys = [
    'uploaded_resumes',
    'prev_uploaded_files',
    'rendered_message_ids',
    'last_user_question',
    'chat_history'
]

for key in state_keys:
    if key not in st.session_state:
        st.session_state[key] = []

for key in ['first_question_processed', 'match_triggered', 'loaded_comparison_table', 'processed_resume_summary']:
    st.session_state[key] = False

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

# Delete already existing resume and section vector db store
delete_faiss_store('faiss_resume_store')
delete_faiss_store('faiss_sections_store')

# -- Streamlit UI logic----

st.markdown("""
<style>
.talai-header {
    position: relative;
    width: 100%;
    height: 220px;
    background-image: url("https://raw.githubusercontent.com/CDSB9GROUP11/TALAI/main/talai-bg1.png");
    background-size: cover;
    background-position: center;
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #111;
    padding: 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.talai-header .logo {
    width: 80px;
    border-radius: 50%;
    box-shadow: 0 0 10px rgba(0,0,0,0.2);
    margin-bottom: 0.5rem;
}

.talai-header h2 {
    margin: 0.2rem 0;
    font-size: 1.5rem;
}

.talai-header p {
    margin: 0;
    font-size: 1.1rem;
}
</style>

<div class="talai-header">
    <img src="https://raw.githubusercontent.com/CDSB9GROUP11/TALAI/main/TalAI.png" width="80" style="border-radius: 50%; margin-bottom: 1rem;">
    <h2 style="color: white; font-weight: bold; font-size: 2rem;">🎖️ TalAI ResumeTools</h2>
    <p style="color: white; font-weight: bold; font-size: 1.2rem;">
        Upload your resume to extract insights, skills, and suggestions.
    </p>
</div>
""", unsafe_allow_html=True)

# --- About Us ---
with st.expander("ℹ️ About TalAI"):
    st.markdown("""
    **TalAI** is a smart, recruiter-centric resume analysis platform designed to deliver precision, clarity, and control at scale. It combines semantic search, metadata tagging, and customizable ranking logic—giving hiring teams to perform below.

    - 📄 Resume parsing and personal information masking  
    - 📊 Ranking 
    - 💬 Chat-based decision panel  
    - 🧠 Summarization and insights  
    """)

# --- Upload UI ---
st.markdown("### 📤 Upload Resumes")
st.markdown("Drag and drop PDF or ZIP files below. ZIPs must contain only PDFs. Limit: 4 pages per file.")

# File uploader
uploaded_files = st.file_uploader(
    "Upload Resumes (PDF, ZIP(containing PDF) allowed)",
    type=["pdf", "zip"],
    accept_multiple_files=True
)

# Reset session state if new files are selected
if uploaded_files != st.session_state.prev_uploaded_files:
    st.session_state.uploaded_resumes = []  # Clear old resumes
    st.session_state.prev_uploaded_files = uploaded_files  # Update tracker

if st.button("Submit and Process") and uploaded_files:
    with st.spinner("🔍 Redacting personal information and preparing files..."):
        for uploaded_file in uploaded_files:
            ext = uploaded_file.name.rsplit('.', 1)[-1].lower()
            if ext == "pdf":
                modified_pdf = redact_personal_information(uploaded_file)
                temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                modified_pdf.seek(0)
                with open(temp_path, "wb") as f:
                    f.write(modified_pdf.read())
                st.session_state.uploaded_resumes.append(Path(temp_path))
            elif ext == "zip":
                zip_bytes = BytesIO(uploaded_file.read())
                extracted_paths = extract_and_mask_from_zip(zip_bytes)
                st.session_state.uploaded_resumes.extend(extracted_paths)

    st.success("✅ Resumes uploaded successfully! All personal information has been securely masked.")

    with st.spinner("🔄 Checking existing resume and section embeddings..."):
        # Extract file and section IDs
        file_ids = [file.stem for file in st.session_state.uploaded_resumes if file.is_file()]
        section_row_ids = [f"{file_id}_{section}" for file_id in file_ids for section in st.session_state.section_list]

        # Fetch existing data - Two scenarios (all present, few present or none present)
        df_resume, new_file_ids = fetch_existing_resumes(st.session_state.database_session, file_ids)
        df_sections, new_section_ids = fetch_existing_resumes(st.session_state.database_session, section_row_ids)

        # Initialize empty resume and sections dataframe for new file ids (if required)
        df_resume_new, df_sections_new = create_vector_storing_dataframes()

        # Check for cases if no resume and sections
        if (df_resume is None and df_sections is None):
            print("Empty and initialization")
            # DataFrames for resume and sections
            df_resume, df_sections = create_vector_storing_dataframes()

        with st.container():
            st.markdown(f"🧾 <span style='font-weight:bold'>{len(file_ids) - len(new_file_ids)} existing resumes</span> detected. Now processing <span style='font-weight:bold'>{len(new_file_ids)} new uploads</span>.", unsafe_allow_html=True)
            st.markdown(f"🧠 <span style='font-weight:bold'>{len(section_row_ids) - len(new_section_ids)} existing sections</span> found. Parsing <span style='font-weight:bold'>{len(new_section_ids)} new sections</span> for embeddings.", unsafe_allow_html=True)

    if new_file_ids:
        with st.spinner(f"⚙️ Processing {len(new_file_ids)} new resumes — generating embeddings for new resumes and sections!"):
            # Initialize progress bar
            progress = st.progress(0)
            status_text = st.empty()

            total = len(new_file_ids)
            for idx, file_id in enumerate(new_file_ids):
                # Match file object by stem
                file = next((f for f in st.session_state.uploaded_resumes if f.stem == file_id), None)
                if not file or not file.is_file():
                    continue

                # Load resume
                single_resume = load_resume_file(str(file), file_id)

                section_prompt_template, json_parser = get_prompt_and_json_parser_for_sections()

                chain = section_prompt_template | st.session_state.llm_model | json_parser

                prompt_output = chain.invoke({"resume": single_resume.page_content})

                # Update section-level vectors
                df_sections_new = update_section_vectors_in_df(
                    st.session_state.embedding_llm,
                    prompt_output,
                    st.session_state.section_list,
                    df_sections_new,
                    file_id,
                    st.session_state.database_session
                )

                # Update resume-level vector
                df_resume_new = update_resume_vector_in_df(
                    st.session_state.embedding_llm,
                    single_resume,
                    df_resume_new,
                    st.session_state.database_session
                )

                # Update progress bar
                progress.progress((idx + 1) / total)
                status_text.text(f"✅ Processed {file_id} ({idx + 1}/{total})")
        st.success("🎉 All new resumes processed and vectorized.")
        status_text.text("✅ Embedding generation complete.")

    df_resume = safe_concat_with_schema(df_resume, df_resume_new)
    df_sections = safe_concat_with_schema(df_sections, df_sections_new)

    section_store, resume_store = build_vector_stores(df_sections, df_resume, st.session_state.embedding_llm)
    st.session_state.index_resume = resume_store
    st.session_state.index_sections = section_store

    st.success("🚀 All resumes and sections have been embedded and now live in the vector database—ready for smart search and matching!")
    # Simulated popup message
    st.toast("Moving to Candidate Search and Screening Tool...", icon="📦")

    # Delay before switching
    time.sleep(2)

    # Switch to next page
    st.switch_page("pages/1_Candidate Search and Screening Tool.py")