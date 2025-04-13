import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 🔌 Import your backend preprocessing logic from the `rudra` folder
from rudra.core import preprocess_data  # Ensure this function exists

# --- Streamlit Config ---
st.set_page_config(page_title="Rudra Auto Data Preprocessing Tool", layout="centered")

# --- App Title ---
st.title("📊 Rudra Auto Data Preprocessing Tool")
st.markdown("Upload your CSV dataset, and let the tool handle preprocessing automatically!")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# --- Optional Preprocessing Options ---
st.sidebar.header("⚙️ Preprocessing Options (Optional)")
remove_outliers = st.sidebar.checkbox("Remove outliers")
scale_data = st.sidebar.checkbox("Scale numeric features")
encode_categoricals = st.sidebar.checkbox("Encode categorical variables")

if uploaded_file is not None:
    try:
        # Read dataset
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Original Dataset")
        st.dataframe(df)

        # --- Visualizations: Original Data ---
        st.markdown("### 📊 Original Data Distributions")
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            fig1 = plt.figure(figsize=(10, 5))
            df[numeric_cols].hist(bins=30)
            st.pyplot(fig1)

        # --- Progress Bar ---
        progress = st.progress(0, text="Starting preprocessing...")
        progress.progress(25, text="Cleaning & transforming data...")

        # --- Call your backend function ---
        processed_df, transformation_summary = preprocess_data(
            df,
            remove_outliers=remove_outliers,
            scale_data=scale_data,
            encode_categoricals=encode_categoricals
        )
        progress.progress(100, text="Preprocessing complete!")

        # --- Show Processed Output ---
        st.subheader("✅ Preprocessed Dataset")
        st.dataframe(processed_df)

        # --- Download Button ---
        csv = processed_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Preprocessed CSV", csv, "processed_data.csv", "text/csv")

        # --- Visualizations: Processed Data ---
        st.markdown("### 🔍 Processed Data Distributions")
        numeric_cols_processed = processed_df.select_dtypes(include='number').columns.tolist()
        if numeric_cols_processed:
            fig2 = plt.figure(figsize=(10, 5))
            processed_df[numeric_cols_processed].hist(bins=30)
            st.pyplot(fig2)

        # --- Transformation Summary ---
        st.markdown("### 📝 Transformation Summary")
        if transformation_summary:
            for step in transformation_summary:
                st.markdown(f"- {step}")
        else:
            st.info("No specific transformation summary provided.")

    except Exception as e:
        st.error(f"⚠️ Error processing the file: {e}")

# --- FAQ Section ---
st.markdown("---")
st.header("❓ Frequently Asked Questions")

faq = {
    "What kind of files can I upload?": "Only CSV files are supported right now.",
    "What preprocessing steps are applied?": "The tool handles missing values, encodes categorical variables, scales numerical values, and optionally removes outliers or performs feature selection, depending on the dataset.",
    "Where is the data processed?": "All data processing happens locally within this app using your custom Python backend.",
    "Can I download the processed data?": "Yes, after preprocessing, you can download the processed dataset as a CSV.",
    "Is my data stored?": "No, your data is not stored anywhere. It's processed temporarily and cleared once you leave or refresh the page."
}

for question, answer in faq.items():
    with st.expander(question):
        st.write(answer)
