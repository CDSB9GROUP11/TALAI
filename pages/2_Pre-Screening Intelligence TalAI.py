from summary_and_section_utilities import *
from nosql_utilities import *
from resume_uploader_utilities import *
from qna_bot import *

# Standard packages
import streamlit as st
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Pre-Screening")

# ------------------ Utility Functions ------------------

def update_selected_index(idx):
    st.session_state.selected_index = idx

def set_selected_index(idx):
    st.session_state.selected_index = idx
    for j in range(len(st.session_state.resume_df)):
        if j != idx:
            st.session_state[f"select_{j}"] = False

def reset_qna():
    st.session_state.chat_history = []
    st.session_state.rendered_message_ids = []
    st.session_state.first_question_processed = False

def stream_response(text):
    container = st.empty()
    full = ""
    for word in text.split():
        full += word + " "
        container.markdown(full)
        time.sleep(0.05)
    return full

def maybe_send_first_question():
    first_question = (
        "Candidate Name, total years of experience, and recent job organisation in bullet points"
    )

    response_text = handle_user_input(
            first_question,
            st.session_state.llm_model,
            st.session_state.embedding_llm,
        )

    st.session_state.chat_history.extend([
        {"role": "user", "content": first_question},
        {"role": "assistant", "content": response_text}
    ])

def get_section_text_by_resume_id(df, resume_id):
    filtered = df[df['resume_id'] == resume_id]
    return filtered['section_text'].tolist()

def render_chat_history():
    scroll_anchor = st.empty()
    logger.info("Chat History")
    logger.info(st.session_state.chat_history)

    for msg in st.session_state.chat_history:
        role = msg["role"]
        icon = "🧑‍💼" if role == "user" else "🤖"
        label = "**You:**" if role == "user" else "**TalAI:**"
        with st.chat_message(role):
            # st.markdown(f"{icon} {label} {msg['content']}")
            if role == "assistant":
                stream_response(f"{icon} {label} {msg['content']}")
            else:
                st.markdown(f"{icon} {label} {msg['content']}")



    scroll_anchor.markdown("")

st.set_page_config("Pre-Screening Intelligence: TalAI", layout="wide")
st.title("🏅 TalAI: Pre-Screening Intelligence Dashboard")

# ─── Top Horizontal Panel ─────────────────────────────────────────────────────


st.subheader("🧑‍💼 Recruiters Decision Panel")

# # ── Triggered Comparison Table ──────────────────────────────────────────────
# if st.session_state.match_triggered and not st.session_state.resume_df.empty:

#     if not st.session_state.loaded_comparison_table:
#         st.session_state.loaded_comparison_table = True

#         resume_ids = st.session_state.resume_df["Resume"].tolist()
#         column_names = ["JD"] + resume_ids

#         selected_sections = (
#             st.session_state.selected_sections + st.session_state.remaining_sections
#             if "selected_sections" in st.session_state and st.session_state.selected_sections
#             else st.session_state.remaining_sections
#         )
#         section_names = [entry['section'] for entry in selected_sections if entry['section'] != 'Entire Resume']

#         section_text_df = fetch_existing_resumes(resume_ids)[0]
#         section_dict_by_resume = {}

#         for resume_label in column_names:
#             content = get_section_text_by_resume_id(section_text_df, resume_label)
#             section_prompt_template, json_parser = get_prompt_and_json_parser_for_comparison_table(is_jd=False)
#             chain = section_prompt_template | mistralai_llm_model() | json_parser
#             section_dict_by_resume[resume_label] = chain.invoke({"resume": content, "jd": st.session_state.job_description})

#         section_prompt_template, json_parser = get_prompt_and_json_parser_for_comparison_table(is_jd=True)
#         chain = section_prompt_template | mistralai_llm_model() | json_parser
#         section_dict_by_resume['JD'] = chain.invoke({"jd": st.session_state.job_description})

#         # Cache for reuse
#         st.session_state.section_dict_by_resume = section_dict_by_resume
#         st.session_state.column_names = column_names
#         st.session_state.section_names = section_names

#     # ── Render Comparison Table ──────────────────────────────────────────────
#     header_cols = st.columns([2] + [3] * len(st.session_state.column_names))
#     header_cols[0].markdown("**🧩 Section**")
#     for i, col in enumerate(header_cols[1:]):
#         col.markdown(f"**{st.session_state.column_names[i]}**")

#     for section in st.session_state.section_names:
#         row_cols = st.columns([2] + [3] * len(st.session_state.column_names))
#         row_cols[0].markdown(f"**🗂️ {section}**")
#         for j, col in enumerate(row_cols[1:]):
#             resume_label = st.session_state.column_names[j]
#             value = st.session_state.section_dict_by_resume[resume_label].get(section, "Section Not Present")

#             if value == "Section Not Present":
#                 col.markdown(f"<span style='color:gray;'>⚠️ <i>Missing</i></span>", unsafe_allow_html=True)
#             elif section == "Skills":
#                 col.markdown(", ".join([
#                     f"<span style='color:#1f77b4;font-weight:bold;'>{skill.strip()}</span>"
#                     for skill in value.split(",")
#                 ]), unsafe_allow_html=True)
#             else:
#                 col.markdown(f"<span style='color:#ff8c00;'>{value}</span>", unsafe_allow_html=True)

#     # ── Review Checkbox Row ──────────────────────────────────────────────────
#     review_cols = st.columns([2] + [3] * len(st.session_state.column_names))
#     review_cols[0].markdown("**🔍 Review**")
#     for i, col in enumerate(review_cols[2:]):
#         resume_label = st.session_state.column_names[i - 1]
#         checked = col.checkbox(
#             label=f"Review {resume_label}",
#             key=f"select_{i}",
#             value=(st.session_state.selected_index == i),
#             label_visibility="collapsed",
#             on_change=set_selected_index,
#             args=(i,)
#         )
#         if checked:
#             st.session_state.selected_index = i
#         elif st.session_state.selected_index == i:
#             st.session_state.selected_index = None

# else:
#     st.info("No resumes are selected. Go to 'Candidate Search and Screening Tool' page to begin.")

if st.session_state.match_triggered and not st.session_state.resume_df.empty:

    # 🔄 Always show dropdown
    resume_ids = st.session_state.resume_df["Resume"].tolist()
    total_resumes = len(resume_ids)
    chunk_size = 3
    range_options = [f"{i}-{min(i + chunk_size, total_resumes)}" for i in range(0, total_resumes, chunk_size)]

    # 🧠 Track dropdown selection in session_state
    if "selected_range" not in st.session_state:
        st.session_state.selected_range = range_options[0]

    selected_range = st.selectbox("📂 Select Resume Range", options=range_options, index=range_options.index(st.session_state.selected_range), key="resume_range_select")

    # 🔁 Detect change and trigger refresh
    if selected_range != st.session_state.selected_range or "section_dict_by_resume" not in st.session_state:
        st.session_state.selected_range = selected_range

        start_idx, end_idx = map(int, selected_range.split("-"))
        selected_resume_ids = resume_ids[start_idx:end_idx]
        column_names = ["JD"] + selected_resume_ids

        st.caption(f"Showing resumes {start_idx + 1} to {end_idx} of {total_resumes}")

        selected_sections = (
            st.session_state.selected_sections + st.session_state.remaining_sections
            if "selected_sections" in st.session_state and st.session_state.selected_sections
            else st.session_state.remaining_sections
        )
        section_names = [entry['section'] for entry in selected_sections if entry['section'] != 'Entire Resume']

        section_text_df = fetch_existing_resumes(selected_resume_ids)[0]
        section_dict_by_resume = {}

        with st.spinner("🔄 Parsing resumes and generating comparison table..."):
            for resume_label in column_names:
                content = get_section_text_by_resume_id(section_text_df, resume_label)
                section_prompt_template, json_parser = get_prompt_and_json_parser_for_comparison_table(is_jd=False)
                chain = section_prompt_template | st.session_state.llm_model | json_parser
                section_dict_by_resume[resume_label] = chain.invoke({"resume": content, "jd": st.session_state.job_description})

            section_prompt_template, json_parser = get_prompt_and_json_parser_for_comparison_table(is_jd=True)
            chain = section_prompt_template | st.session_state.llm_model | json_parser
            section_dict_by_resume['JD'] = chain.invoke({"jd": st.session_state.job_description})

        # 🧠 Cache results
        st.session_state.section_dict_by_resume = section_dict_by_resume
        st.session_state.column_names = column_names
        st.session_state.section_names = section_names

    # ── Render Comparison Table ──────────────────────────────────────────────
    header_cols = st.columns([2] + [3] * len(st.session_state.column_names))
    header_cols[0].markdown("**🧩 Section**")
    for i, col in enumerate(header_cols[1:]):
        col.markdown(f"**{st.session_state.column_names[i]}**")

    for section in st.session_state.section_names:
        row_cols = st.columns([2] + [3] * len(st.session_state.column_names))
        row_cols[0].markdown(f"**🗂️ {section}**")
        for j, col in enumerate(row_cols[1:]):
            resume_label = st.session_state.column_names[j]
            value = st.session_state.section_dict_by_resume[resume_label].get(section, "Section Not Present")

            if value == "Section Not Present":
                col.markdown(f"<span style='color:gray;'>⚠️ <i>Missing</i></span>", unsafe_allow_html=True)
            elif section == "Skills":
                # col.markdown(", ".join([
                #     f"<span style='color:#1f77b4;font-weight:bold;'>{skill.strip()}</span>"
                #     for skill in value.split(",")
                # ]), unsafe_allow_html=True)
                logger.info(f"Skills: {value}")
                # 🧹 Normalize and split on 'Missing:'
                if "Missing:" in value:
                    present_part, missing_part = value.split("Missing", 1)
                    present_skills = list(set([skill.strip() for skill in present_part.split(",") if skill.strip()]))
                    missing_skills = list(set([skill.strip() for skill in missing_part.split(",") if skill.strip()]))
                else:
                    present_skills = list(set([skill.strip() for skill in value.split(",") if skill.strip()]))
                    missing_skills = []

                # 🎨 Apply red styling to missing skills
                styled_skills = present_skills + [
                    f"<span style='color:#ff0000;'>{skill}</span>" for skill in missing_skills
                ]

                col.markdown(", ".join(styled_skills), unsafe_allow_html=True)
            else:
                col.markdown(f"<span style='color:#ff8c00;'>{value}</span>", unsafe_allow_html=True)

    # ── Review Checkbox Row ──────────────────────────────────────────────────
    review_cols = st.columns([2] + [3] * len(st.session_state.column_names))
    review_cols[0].markdown("**🔍 Review**")
    for i, col in enumerate(review_cols[2:]):
        print(f"Resume Label: {resume_label}, Index: {i}")
        resume_label = st.session_state.column_names[i+1]
        checked = col.checkbox(
            label=f"Review {resume_label}",
            key=f"select_{i}",
            value=(st.session_state.selected_index == i),
            label_visibility="visible"
        )
        if checked:
            st.session_state.selected_index = i
        elif st.session_state.selected_index == i:
            st.session_state.selected_index = None

else:
    st.info("No resumes are selected. Go to 'Candidate Search and Screening Tool' page to begin.")


# ─── Bottom Horizontal Panel ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🔍 Resume & Video Summarization Dashboard")

left_col, right_col = st.columns([2, 2])

# ── Left Vertical Panel: Resume Summarization ────────────────────────────────
with left_col:
    if st.session_state.selected_index is not None:
        selected_row = st.session_state.resume_df.loc[st.session_state.selected_index]
        resume_id = selected_row["Resume"]

        # Only regenerate summary if selection changed
        if "previous_selected_index" not in st.session_state:
            st.session_state.previous_selected_index = -1

        if st.session_state.selected_index != st.session_state.previous_selected_index:
            st.session_state.resume_summary_content = None
            with st.spinner("🔄 Generating summarization and embeddings..."):
                selected_df, _ = fetch_existing_resumes([resume_id])
                st.session_state.selected_resume = selected_df
                resume_content = selected_df['section_text'].iloc[0] if 'section_text' in selected_df.columns and not selected_df.empty else ''
                st.session_state.resume_summary_content = resume_summarization(resume_content, st.session_state.openai_llm_model)
            st.session_state.previous_selected_index = st.session_state.selected_index

        # Display summarization
        st.subheader("📝 Summarization Dashboard")
        st.markdown(st.session_state.resume_summary_content)

    else:
        st.info("Select a resume from the comparison table above to view its summary.")

# ── Right Vertical Panel: Video Summary + TalAI Chat ─────────────────────────
with right_col:
    st.subheader("🎥 Video Summary")
    st.button("Summarize Video")
    st.write("Video summary will appear here.")
    st.image(
        "/content/drive/MyDrive/CDS-B9-Group11/Capstone Project/video_image.png",
        caption="Video Preview Placeholder",
        width="stretch"
    )
    with st.popover("💬 Chat with TalAI", icon=":material/chat:", width="content"):
        if (
            st.session_state.selected_index is not None
            and not st.session_state.resume_df.empty
        ):
            resume_id = st.session_state.resume_df.loc[st.session_state.selected_index]["Resume"]

            if st.session_state.get("resume_summary_content") and resume_id not in st.session_state.processed_resumes:
                reset_qna()
                resume_content = st.session_state.selected_resume['section_text'].iloc[0] if 'section_text' in st.session_state.selected_resume.columns and not st.session_state.selected_resume.empty else "Section text not found"
                text_chunks = get_text_chunks(resume_content)
                vector_store_using_chunks(text_chunks, st.session_state.embedding_llm)
                maybe_send_first_question()
                st.session_state.processed_resumes.add(resume_id)

            user_question = st.chat_input("Ask TalAI about the selected resume...")

            if user_question:
                with st.chat_message("user"):
                    st.markdown(f"🧑‍💼 **You:** {user_question}")
                with st.chat_message("assistant"):
                    response = handle_user_input(user_question, st.session_state.llm_model, st.session_state.embedding_llm)
                    full_response = stream_response(response)
                st.session_state.chat_history.extend([
                    {"role": "user", "content": user_question},
                    {"role": "assistant", "content": response}
                ])
            render_chat_history()

            # Optional: Auto-scroll to bottom
            st.markdown("""
            <div id="scroll-to-bottom"></div>
            <script>
                var el = document.getElementById("scroll-to-bottom");
                if (el) el.scrollIntoView({behavior: "smooth"});
            </script>
            """, unsafe_allow_html=True)
        else:
            st.info("Select a resume to start chatting with TalAI.")