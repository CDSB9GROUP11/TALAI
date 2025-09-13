##user defined packages and libraries import
from resume_uploader_utilities import *

#Standard packages
import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="Candidate Search & Screening Tool", layout="wide")
st.title("🎯 Candidate Search & Screening Tool")

# Initialize session state variable
if "uploaded_resumes" not in st.session_state or not st.session_state.uploaded_resumes:
    st.info("No resumes uploaded yet. Go to 'Resume Uploader' page to begin.")
else:
    count = len(st.session_state.uploaded_resumes)
    st.subheader(f"📄 {count} resume{'s' if count != 1 else ''} uploaded — ready for smart matching")

# --- Job Description Input ---
st.subheader("Job Description")
job_description = st.text_area(
    "📝 Job Description",
    height=200,
    placeholder="Paste the job role or requirements here..."
)
st.session_state.job_description = job_description

# --- Dropdowns ---
col1, col2 = st.columns(2)
with col1:
    job_area = st.selectbox("Domain",
                            ["Information Technology", "Finance", "Human Resources", "Sales", "Legal/Advocate", "Engineering"])
with col2:
    top_n = st.number_input("Number of Profiles to show", min_value=2, max_value=20, value=2, step=1)
    st.session_state.matched_resume_count = top_n

st.markdown("### 🧑‍💼 Recruiters Input Dashboard")

# Custom sort order
section_order = [
    "Work Experience",
    "Skills",
    "Education",
    "Certifications, Awards and Recognitions",
    "Entire Resume",
    "Internship and Academic Projects"
]

# Default weight values
default_weights = [30, 25, 20, 10, 9, 6]

# Initialize session state
if "selected_sections" not in st.session_state:
    st.session_state.selected_sections = []

# Initialize remaining section state
if "remaining_sections" not in st.session_state:
    st.session_state.remaining_sections = []

# Full list of available sections
all_sections = get_section_headings()
all_sections.insert(0, "Entire Resume")

# ── Render Section Dropdowns and Weights ───────────────────────────────────
remove_idx = None  # Track which index to remove

# Compute remaining sections
selected_names = [entry["section"] for entry in st.session_state.selected_sections if entry["section"]]

# Compute remaining sections based on priority
remaining = [s for s in section_order if s not in selected_names]

# Sort remaining sections by priority
sorted_remaining = sorted(remaining, key=lambda x: section_order.index(x))

# Get last N weights from default_weights based on how many remaining sections we have
remaining_weights = default_weights[-len(sorted_remaining):] if sorted_remaining else []

# Assign weights to remaining sections
st.session_state.remaining_sections = [
    {"section": sec, "weight": wt} for sec, wt in zip(sorted_remaining, remaining_weights)
]

# Display remaining sections with weights
st.subheader("🧾 Default Weights for Pre-Defined Sections")

for entry in st.session_state.remaining_sections:
    st.markdown(f"- **{entry['section']}** → `{entry['weight']}`")

st.subheader("⚖️ Add dynamic sections to reassign weights for improved accuracy based on job description and search relevance.")

# ── Add Section Button ─────────────────────────────────────────────────────
if st.button("➕ Add Section"):
    default_weights = [30, 25, 20, 10, 9, 6]
    weight = default_weights[len(st.session_state.selected_sections)] if len(st.session_state.selected_sections) < len(default_weights) else 0
    st.session_state.selected_sections.append({"section": None, "weight": weight})

for idx, entry in enumerate(st.session_state.selected_sections):
    selected = entry["section"]
    already_selected = [e["section"] for i, e in enumerate(st.session_state.selected_sections) if i != idx]
    available_options = [s for s in all_sections if s not in already_selected]

    cols = st.columns([2, 1, 1])

    with cols[0]:
        # 🛡️ Safe index fallback
        default_index = available_options.index(selected) if selected in available_options else 0
        section = st.selectbox(
            f"Section {idx+1}",
            options=available_options,
            index=default_index,
            key=f"section_{idx}"
        )
        st.session_state.selected_sections[idx]["section"] = section

    with cols[1]:
        # 🎯 Display weight as static text
        st.markdown(f"**Weight:** {entry['weight']}")

    with cols[2]:
        if st.button("❌ Remove", key=f"remove_{idx}"):
            remove_idx = idx

# 🔧 Remove outside loop to avoid layout glitches
if remove_idx is not None:
    st.session_state.selected_sections.pop(remove_idx)

#Prepare weight by section
weightage_by_section = {}
for sec in st.session_state.selected_sections + st.session_state.remaining_sections:
    weightage_by_section[sec['section']] = float(sec['weight'])

# Submit button
if st.button("🔍 Search Candidates") and job_description and st.session_state.uploaded_resumes:
    st.session_state.comparison_table = None #Reset the tabel

    # Create dataframes for storing similarity scores (ID, Section Title and Similarity Score)
    df_resume_similarity_score,df_sectionwise_similarity_score = create_similarity_score_dataframes()

    # Get the similarity score from resume index and select top n*2 resumes
    #Create embeddings for provided job description
    job_description_embedding = get_vector_embeddings(job_description, st.session_state.embedding_llm)

    n = top_n*2

    #Get resume similarity score
    df_resume_similarity_score = coarse_filter_resumes_faiss(
        st.session_state.index_resume,
        job_description_embedding,
        n,
        df_resume_similarity_score)

    # Get the sectionwise similarity score
    resume_id_list = df_resume_similarity_score.id.tolist()

    df_sectionwise_similarity_score = compute_sectionwise_similarity_scores_faiss(
        st.session_state.index_sections,
        job_description_embedding,
        resume_id_list,
        df_sectionwise_similarity_score)

    # combine 2 dataframes
    df_combined_similarity_score = pd.concat([df_resume_similarity_score, df_sectionwise_similarity_score], ignore_index=True)

    # create a dataframe to store the weighted sum
    df_candidate_weighted_score = pd.DataFrame(columns=["id","weighted_sum_of_similarity_scores"])

    # Calculate final weighted similarity score and select top n resumes

    df_candidate_weighted_score = calculate_weighted_combined_similarity_score (resume_id_list, df_combined_similarity_score, weightage_by_section, df_candidate_weighted_score)

    df_candidate_weighted_score.sort_values(by='weighted_sum_of_similarity_scores', ascending=False, inplace=True)

    st.session_state.resume_df = convert_to_ranked_df(df_candidate_weighted_score.head(top_n))

    logger.info(st.session_state.resume_df)

    st.session_state.match_triggered = True
    st.session_state.matched_resume_count = top_n
    st.success("Top resumes matched and stored.")

    # Simulated popup message
    st.toast("Moving to Candidate Search and Screening Tool...", icon="📦")

    # Delay before switching
    time.sleep(2)

    # Switch to next page
    st.switch_page("2_Pre-Screening Intelligence TalAI.py")