import streamlit as st
import time
from dotenv import load_dotenv
import os

# Import your existing app methods
from app import load_and_index_transcript, answer_question

# Load environment variables
load_dotenv()

# --- Streamlit page config ---
st.set_page_config(
    page_title="YouTube Video Q&A Assistant",
    layout="wide"
)
st.title("YouTube Video Q&A Assistant")
st.caption("Ask questions about any YouTube video by loading its transcript!")

# --- Session state initialization ---
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- Sidebar: Load transcript ---
with st.sidebar:
    st.header("Load YouTube Transcript")
    video_id = st.text_input("YouTube Video ID", value=st.session_state.video_id or "")
    
    if st.button("Load Transcript"):
        if not video_id.strip():
            st.error("Please enter a valid YouTube video ID.")
        else:
            with st.spinner("Fetching and indexing transcript..."):
                try:
                    vectorstore = load_and_index_transcript(video_id)
                    st.session_state.video_id = video_id
                    st.session_state.vectorstore = vectorstore
                    st.success("Transcript loaded successfully!")
                except Exception as e:
                    st.error(f"Failed to load transcript: {e}")

# --- Main content (inside your existing file) ---
st.subheader("Ask a Question")

if not st.session_state.get("vectorstore"):
    st.info("Please load a YouTube transcript first using the sidebar.")
else:
    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if not question or not question.strip():
            st.warning("Please enter a valid question.")
        else:
            # Ensure video_id is present
            vid = st.session_state.get("video_id")
            if not vid:
                st.error("No video ID found. Please load transcript again.")
            else:
                with st.spinner("Generating answer..."):
                    try:
                        # <-- IMPORTANT: pass both video_id and question
                        full_answer = answer_question(vid, question.strip())

                        # Stream the answer (word-by-word)
                        response_placeholder = st.empty()
                        streamed_text = ""
                        for word in full_answer.split():
                            streamed_text += word + " "
                            response_placeholder.markdown(f"**Answer:** {streamed_text}▌")
                            time.sleep(0.3)
                        response_placeholder.markdown(f"**Answer:** {streamed_text}")

                    except Exception as e:
                        st.error(f"Error while generating answer: {e}")


# --- Footer ---
st.markdown("---")
st.markdown("*Built with LangChain + HuggingFace + Google Flan-T5 + ChromaDB + Streamlit*")
