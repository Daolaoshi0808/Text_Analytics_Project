import streamlit as st
from model_runner import predict_case_attributes
from legal_summary import legal_summary_pipeline

st.set_page_config(page_title="Legal Case Analyzer", layout="wide")
st.title("📄 Legal Case Classifier + Summarizer")

st.markdown("""
Enter a legal case and:
- 🔍 Predict if it's a **class action**
- 🧠 Identify the **case group type**
- 📝 Optionally generate a **summary**

---
""")

input_mode = st.radio("Choose input method:", ["Paste Text", "Upload .txt File"])
user_input = ""

if input_mode == "Paste Text":
    user_input = st.text_area("Paste full legal case text:", height=300)
elif input_mode == "Upload .txt File":
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    if uploaded_file:
        user_input = uploaded_file.read().decode("utf-8")
        st.text_area("File content:", user_input, height=300, disabled=True)

# Let user optionally select a summary type
summary_style = st.selectbox("Optional: Generate a summary", ["None", "long", "short", "tiny"])

if st.button("Run Analysis"):
    if not user_input.strip():
        st.warning("Please enter or upload some text.")
    else:
        with st.spinner("Running predictions..."):
            prediction = predict_case_attributes(user_input)

        st.success("Prediction complete!")
        st.markdown(f"**Class Action Sought**: `{prediction['Class Action Sought']}`")
        st.markdown(f"**Predicted Case Group**: `{prediction['Predicted Case Group']}`")

        if summary_style != "None":
            with st.spinner(f"Generating {summary_style} summary..."):
                summary = legal_summary_pipeline(user_input, summary_style=summary_style)
            st.subheader("📚 Summary")
            st.text_area("Summary Output", summary, height=200, disabled=True)

        st.markdown("---")
        st.subheader("🔍 Input Preview")
        st.text_area("Your Input", user_input, height=150, disabled=True)
