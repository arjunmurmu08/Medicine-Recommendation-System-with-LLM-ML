import streamlit as st
import os
import sys
from openai import OpenAI

# --- BACKEND IMPORT ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from model_api import DiseaseRecommender


# --- OPENAI API KEY AND CLIENT ---
OPENAI_API_KEY = st.secrets["OPENAI"]["API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

# --- INIT RECOMMENDER ---
recommender = DiseaseRecommender()

# --- SESSION STATE SETUP ---
if "diagnosis" not in st.session_state:
    st.session_state["diagnosis"] = None
if "details" not in st.session_state:
    st.session_state["details"] = None
if "llm_answer" not in st.session_state:
    st.session_state["llm_answer"] = None
if "selected_symptoms" not in st.session_state:
    st.session_state["selected_symptoms"] = []

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title="LLM Medicine Recommender", page_icon="🤖", layout="centered")
st.title("🩺 Disease Detection & Medicine Recommendation (with LLM Details)")
st.write("Select your symptoms to receive a model prediction, medical suggestions, and AI explanations.")

all_symptoms = list(recommender.symptoms_dict.keys())

# --- Symptoms selection ---
st.session_state["selected_symptoms"] = st.multiselect(
    "Select your symptoms:", all_symptoms, default=st.session_state["selected_symptoms"]
)

# --- Diagnose Button ---
if st.button("Diagnose"):
    if not st.session_state["selected_symptoms"]:
        st.warning("Please select at least one symptom.")
    else:
        disease = recommender.predict(st.session_state["selected_symptoms"])
        details = recommender.get_details(disease)
        st.session_state["diagnosis"] = disease
        st.session_state["details"] = details
        st.session_state["llm_answer"] = None   # reset explanation

# --- IF DIAGNOSIS EXISTS, SHOW DETAILS/LLM UI ---
if st.session_state["diagnosis"]:
    disease = st.session_state["diagnosis"]
    details = st.session_state["details"]

    st.header(f"Prediction: {disease}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Description:**")
        st.info(details["description"] or "No description found.")
        st.markdown("**Medications:**")
        for med in details["medications"]:
            st.write(f"• {med}")
        st.markdown("**Diet:**")
        for d in details["diets"]:
            st.write(f"• {d}")
    with col2:
        st.markdown("**Precautions:**")
        for p in details["precautions"]:
            st.write(f"• {p}")
        st.markdown("**Workouts:**")
        for w in details["workouts"]:
            st.write(f"• {w}")

    st.subheader("💡 Ask a question or get a detailed medical explanation:")
    user_q = st.text_area(
        "Your question (or leave blank for summary):",
        key="llm_query"
    )

    if st.button("Get AI Explanation"):
        model_summary = (
            f"Symptoms: {', '.join(st.session_state['selected_symptoms'])}\n"
            f"Disease: {disease}\n"
            f"Description: {details['description']}\n"
            f"Precautions: {', '.join(details['precautions'])}\n"
            f"Medications: {', '.join(details['medications'])}\n"
            f"Diets: {', '.join(details['diets'])}\n"
            f"Workouts: {', '.join(details['workouts'])}\n"
        )
        base_prompt = (
            "You are a compassionate medical assistant. "
            "Provide an in-depth but friendly explanation about the diagnosed disease, "
            "covering what it is, why these medications and precautions are useful, "
            "diet/lifestyle tips, and any important information. "
            "Do not make a diagnosis or suggest drugs outside of the provided list. "
        )
        if user_q.strip():
            full_message = model_summary + f"\nPatient question: {user_q.strip()}"
        else:
            full_message = model_summary + "\nSummarize the above for the patient in detail."

        with st.spinner("Getting detailed medical info..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": base_prompt},
                        {"role": "user", "content": full_message}
                    ],
                    temperature=0.7,
                    max_tokens=768,
                )
                st.session_state["llm_answer"] = response.choices[0].message.content
            except Exception as e:
                st.session_state["llm_answer"] = f"Error: {e}"

    # Show LLM answer if present
    if st.session_state["llm_answer"]:
        st.markdown("#### AI-Powered Explanation")
        st.success(st.session_state["llm_answer"])

st.caption("Powered by SVM + OpenAI LLM. Developed for personalized medicine support.")
