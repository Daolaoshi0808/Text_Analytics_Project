import streamlit as st
from model_runner import predict_case_attributes

st.set_page_config(page_title="Legal Case Analyzer", layout="wide")

st.title("📄 Legal Case Classifier")
st.markdown("""
Enter a full legal case (real or simulated), and this tool will:
- Predict if a **class action** was sought
- Identify the **case group type** (e.g., Civil Rights, Social Welfare)

*Note: Summarization coming soon.*
""")

# Input option: text or file
input_mode = st.radio("Choose input method:", ["Paste Text", "Upload .txt File"])
user_input = ""

if input_mode == "Paste Text":
    def_text = """This action is brought on behalf of a class of employees who have been systematically denied compensation..."""
    user_input = st.text_area("Paste full legal case text:", height=300, value=def_text)

elif input_mode == "Upload .txt File":
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    if uploaded_file is not None:
        user_input = uploaded_file.read().decode("utf-8")
        st.text_area("File content:", user_input, height=300, disabled=True)

if st.button("🔍 Analyze Case"):
    if user_input.strip() == "":
        st.warning("Please enter or upload some text for analysis.")
    else:
        with st.spinner("Classifying case..."):
            prediction = predict_case_attributes(user_input)

        st.success("Prediction complete!")

        st.markdown("### 🧾 Results")
        st.markdown(f"**Class Action Sought**: `{prediction['Class Action Sought']}`")
        st.markdown(f"**Predicted Case Group**: `{prediction['Predicted Case Group']}`")

        st.markdown("---")
        st.subheader("🔍 Input Preview")
        st.text_area("Your Input (Read-Only)", user_input, height=150, disabled=True)
