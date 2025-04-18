import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 🔌 Import backend preprocessing logic
from rudra.core import preprocess_data
from rudra.executor import run_pipeline
from rudra.null_handler import handle_missing_values
from rudra.encoding import encode_categoricals_method
from rudra.preprocess_tree_based import tree_based_preprocess

# --- Streamlit Config ---
st.set_page_config(page_title="Rudra Auto Data Preprocessing Tool", layout="centered")

# --- App Title ---
st.title("📊 Rudra Auto Data Preprocessing Tool")
st.markdown("Upload your CSV dataset, and let the tool handle preprocessing based on your selections.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# --- Sidebar Options ---
st.sidebar.header("⚙️ Preprocessing Options")

# ✅ Dropdown for missing value handling
missing_value_strategy = st.sidebar.selectbox(
    "🧼 Missing Value Strategy",
    [
        "None",
        "Drop rows with any missing values",
        "Drop rows with >50% nulls, impute rest (mean/mode)"
    ]
)

remove_outliers = st.sidebar.checkbox("Remove outliers")
scale_data = st.sidebar.checkbox("Scale numeric features")
encode_categoricals = st.sidebar.checkbox("Encode categorical variables")
feature_selection = st.sidebar.checkbox("Perform Feature Selection (Correlation Filtering)")
data_balancing = st.sidebar.checkbox("Perform Data Balancing")

# --- Main Logic ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.warning("⚠️ Uploaded CSV is empty.")
        else:
            st.subheader("📄 Original Dataset")
            st.dataframe(df)

            summary = []
            preprocessed_df = df.copy()

            # ✅ Apply missing value handling only if strategy is selected
            if missing_value_strategy == "Drop rows with any missing values":
                preprocessed_df = preprocessed_df.dropna()
                summary.append("Dropped all rows with any missing values.")
            elif missing_value_strategy == "Drop rows with >50% nulls, impute rest (mean/mode)":
                preprocessed_df, impute_summary = handle_missing_values(preprocessed_df)
                summary += impute_summary

            # ✅ Apply encoding if selected
            if encode_categoricals:
                preprocessed_df = encode_categoricals_method(preprocessed_df)
                summary.append("Encoded categorical variables using Label Encoding.")

            # ✅ Apply feature selection and balancing if selected
            if feature_selection or data_balancing:
                preprocessed_df = tree_based_preprocess(
                    preprocessed_df,
                    target="target" if "target" in preprocessed_df.columns else None,
                    balance_method="class_weight" if data_balancing else "none"
                )
                summary.append("Applied tree-based preprocessing (Feature selection &/or Data balancing).")

            # ✅ Run general preprocessing pipeline
            if remove_outliers or scale_data:
                preprocessed_df, pipeline_summary = run_pipeline(
                    preprocessed_df,
                    remove_outliers=remove_outliers,
                    scale_data=scale_data,
                    encode_categoricals=False  # Already encoded if chosen
                )
                summary += pipeline_summary

            st.markdown("---")
            st.subheader("📋 Preprocessed Dataset")
            st.dataframe(preprocessed_df)

            # --- Download Button ---
            csv = preprocessed_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Preprocessed CSV", csv, "processed_data.csv", "text/csv")

            # --- Transformation Summary ---
            st.markdown("### 🗒️ Transformation Summary")
            if summary:
                for step in summary:
                    st.markdown(f"- {step}")
            else:
                st.info("No preprocessing applied yet. Select options from the sidebar to begin.")

    except Exception as e:
        st.error("⚠️ Error processing the file.")
        st.exception(e)

# --- FAQ Section ---
st.markdown("---")
st.header("❓ Frequently Asked Questions")

faq = {
    "What kind of files can I upload?": "Only CSV files are supported right now.",
    "What preprocessing steps are applied?": "The tool handles missing values, encodes categorical variables, scales numerical values, removes outliers, performs feature selection, and balances the data if needed, depending on the dataset and user selections.",
    "Where is the data processed?": "All data processing happens locally within this app using your custom Python backend.",
    "Can I download the processed data?": "Yes, after preprocessing, you can download the processed dataset as a CSV.",
    "Is my data stored?": "No, your data is not stored anywhere. It's processed temporarily and cleared once you leave or refresh the page."
}

for question, answer in faq.items():
    with st.expander(question):
        st.write(answer)
