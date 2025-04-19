# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io # Required for capturing print outputs
from contextlib import redirect_stdout # Required for capturing print outputs
import smtplib # For sending email
import ssl     # For secure connection
from email.message import EmailMessage # For creating email
import re # For basic email validation

# --- Import Rudra Preprocessing Classes ---
# Assuming 'rudra' package is accessible
try:
    from rudra import PreprocessTreeBased, PreprocessDistanceBased, PreprocessRegressionBased
    # Import others as they are built, e.g.:
    # from rudra.api.client import PreprocessAPI
except ImportError as e:
    st.error(f"Error importing Rudra library: {e}. Make sure 'rudra' is installed or in the correct path.")
    st.stop() # Stop execution if library isn't found


# --- Page Configuration (Optional but Recommended) ---
st.set_page_config(
    page_title="RUDRA ML Preprocessor",
    page_icon="✨",
    layout="wide", # Use wide layout for better dataframe display
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
# Use session state to store uploaded data, processed data, fitted processor, etc.
# This prevents losing state when widgets are interacted with.
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'fitted_processor' not in st.session_state:
    st.session_state.fitted_processor = None
if 'processing_summary' not in st.session_state:
    st.session_state.processing_summary = None
if 'last_params' not in st.session_state:
    st.session_state.last_params = {} # Store params used for last run

# --- Helper Function to Load Data ---
def load_data(uploaded_file):
    try:
        # Infer compression based on extension - more robust
        df = pd.read_csv(uploaded_file, compression='infer')
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# --- Helper Function to Capture Print Output ---
def capture_processing_output(func, *args, **kwargs):
    """Runs a function and captures its print output."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = func(*args, **kwargs) # Execute the function (e.g., processor.fit)
    summary = buffer.getvalue()
    return result, summary

# --- Sidebar ---
with st.sidebar:
    st.image("assets/rudra_logo.png", width=64) # Placeholder icon
    st.title("RUDRA Setup")
    st.markdown("Upload your dataset and choose a preprocessing family.")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv", "gz", "zip"], # Allow compressed files
        key="file_uploader",
        help="Upload your dataset in CSV format (can be compressed)."
    )

    # Process uploaded file only once or if it changes
    if uploaded_file is not None:
        # Check if it's a new file or the first upload
        if st.session_state.original_df is None or uploaded_file.file_id != st.session_state.get('last_file_id', None):
            with st.spinner("Loading data..."):
                df = load_data(uploaded_file)
                if df is not None:
                    st.session_state.original_df = df
                    # Line ~81 in app.py
                    st.session_state.last_file_id = uploaded_file.file_id
                    # Clear previous results when new data is loaded
                    st.session_state.processed_df = None
                    st.session_state.fitted_processor = None
                    st.session_state.processing_summary = None
                    st.session_state.last_params = {}
                    st.success("Data loaded successfully!")
                else:
                    # If loading failed, reset state
                    st.session_state.original_df = None
                    st.session_state.last_file_id = None

    # Display preview if data is loaded
    if st.session_state.original_df is not None:
        st.divider()
        st.subheader("Data Preview")
        st.dataframe(st.session_state.original_df.head(5), height=200) # Show small preview
        st.write(f"Shape: {st.session_state.original_df.shape}")
        st.divider()

        # --- Select Preprocessing Family ---
        st.subheader("Preprocessing Type")
        families = ["Tree-Based", "Distance-Based", "Regression-Based"] # Add "API-Based" when ready
        family_choice = st.radio(
            "Select the model family you intend to use:",
            families,
            key="family_selector"
        )
        st.session_state.selected_family = family_choice
    else:
        st.info("Upload a dataset to configure preprocessing.")


# --- Main Page Content ---

# --- Helper function for centering ---
def centered_html(html_content):
    st.markdown(f"<div style='text-align: center;'>{html_content}</div>", unsafe_allow_html=True)

# --- Main Page Content ---
if st.session_state.original_df is None:
    # Main Welcome Area (Centered using helper)
    centered_html("<h1>Welcome to the RUDRA ML Preprocessor ✨</h1>") # Use H1 for main title
    centered_html("<h3>Effortlessly Prepare Your Data for Machine Learning</h3>") # Use H3 for subtitle
    st.info("👈 **Upload your CSV dataset using the sidebar to begin!**", icon="⬆️") # Bolding here works
    st.divider()

    # --- Tabs for Information ---
    tab_overview, tab_how, tab_features, tab_contact = st.tabs([
        "🎯 Overview",
        "⚙️ How It Works",
        "🚀 Features",
        "✉️ Contact"
    ])

    # --- Overview Tab ---
    with tab_overview:
        # Center header using specific markdown/HTML
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #1f1c2c, #928dab);
                padding: 2rem;
                border-radius: 12px;
                color: #f0f0f5;
                max-width: auto;
                margin: auto;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            ">
            <h2 style="text-align:center; color: #FACC15; margin-bottom: 0.5rem;">
                ✨ About RUDRA
            </h2>
            <p style="text-align:center; color: #DDD; font-size:1.1rem; margin-bottom: 2rem;">
                Tackling data preprocessing can be a slog—RUDRA automates it, so you can focus on modeling!
            </p>

            <div style="
                display: flex;
                flex-wrap: wrap;
                gap: 1.5rem;
                justify-content: space-around;
            ">
            <div style="flex:1; min-width:200px;">
            <h3 style="color:#34D399; margin-bottom:0.3rem;">🌳 Tree‑Based</h3>
            <p style="color:#EEE; margin:0;">
                Decision Trees, Random Forests, XGBoost, LightGBM, CatBoost:
                smart categorical encoding, scaling skipped where not needed.
            </p>
            </div>

            <div style="flex:1; min-width:200px;">
            <h3 style="color:#60A5FA; margin-bottom:0.3rem;">📏 Distance‑Based</h3>
            <p style="color:#EEE; margin:0;">
                KNN, SVM, Clustering:
                robust scaling & encoding to keep your distance metrics honest.
            </p>
            </div>

            <div style="flex:1; min-width:200px;">
            <h3 style="color:#F472B6; margin-bottom:0.3rem;">📈 Regression‑Based</h3>
            <p style="color:#EEE; margin:0;">
                Linear/Logistic Regression, Ridge, Lasso, SVR:
                both encoding and scaling tuned for regression stability.
            </p>
            </div>
            </div>

            <hr style="border-color:#333; margin:2rem 0;" />

            <p style="text-align:center; color:#CCC; font-size:0.95rem;">
                **Key Features:** missing‑value imputation • categorical encoding (One‑Hot, Label) •
                outlier detection & clipping • feature scaling (Standard, MinMax) •
                detailed summary logs • downloadable CSV results.
            </p>

            <p style="text-align:center; margin-top:1.5rem;">
                <a href="https://github.com/bruhat15/Rudra" 
                style="
                    background: #FACC15;
                    color: #111;
                    padding: 0.6rem 1.2rem;
                    border-radius: 6px;
                    text-decoration: none;
                    font-weight: bold;
                ">
                🚀 View on GitHub
                </a>
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- How It Works Tab (Flowchart Simulation) ---
    with tab_how:
        st.markdown("<h2 style='text-align: center;'>Your Preprocessing Workflow</h2>", unsafe_allow_html=True)
        st.write("") # Add some vertical space

        # Using columns implies visual separation and centering effect
        col1, col2, col3, col4, col5 = st.columns(5)
        step_style = "<div style='text-align: center; font-weight: bold;'>{}</div>" # Style for step titles
        arrow_style = "<div style='text-align: center; font-size: 1.5em; margin-top: 5px; margin-bottom: 5px;'>{}</div>" # Style for arrows
        desc_style = "<small><div style='text-align: center;'>{}</div></small>" # Style for descriptions

        with col1:
            st.markdown(step_style.format("① Upload"), unsafe_allow_html=True)
            st.markdown(arrow_style.format("📄➡️"), unsafe_allow_html=True)
            st.markdown(desc_style.format("(Use Sidebar)"), unsafe_allow_html=True)
            st.markdown(desc_style.format("CSV / Compressed"), unsafe_allow_html=True)

        with col2:
            st.markdown(step_style.format("② Select Family"), unsafe_allow_html=True)
            st.markdown(arrow_style.format("👨‍👩‍👧‍👦➡️"), unsafe_allow_html=True)
            st.markdown(desc_style.format("(Use Sidebar)"), unsafe_allow_html=True)
            st.markdown(desc_style.format("Tree, Distance, Regression"), unsafe_allow_html=True)

        with col3:
            st.markdown(step_style.format("③ Configure"), unsafe_allow_html=True)
            st.markdown(arrow_style.format("🔧➡️"), unsafe_allow_html=True)
            st.markdown(desc_style.format("(Main Area)"), unsafe_allow_html=True)
            st.markdown(desc_style.format("Options & Target Model"), unsafe_allow_html=True)

        with col4:
            st.markdown(step_style.format("④ Process"), unsafe_allow_html=True)
            st.markdown(arrow_style.format("⚡️➡️"), unsafe_allow_html=True)
            st.markdown(desc_style.format("(Main Area)"), unsafe_allow_html=True)
            st.markdown(desc_style.format("Run Pipeline"), unsafe_allow_html=True)

        with col5:
            st.markdown(step_style.format("⑤ Review & Download"), unsafe_allow_html=True)
            st.markdown(arrow_style.format("📊💾"), unsafe_allow_html=True)
            st.markdown(desc_style.format("(Main Area)"), unsafe_allow_html=True)
            st.markdown(desc_style.format("Data, Summary, CSV"), unsafe_allow_html=True)

        st.write("")
        st.success("**Follow these simple steps to get clean, consistent data ready for your models!**") # Bolding here

    # --- Features Tab ---
    with tab_features:
        st.markdown("<h2 style='text-align: center;'>Core Capabilities</h2>", unsafe_allow_html=True)
        st.write("")
        col_feat1, col_feat2 = st.columns(2)
        with col_feat1:
            st.subheader("🧹 Data Cleaning & Preparation")
            st.markdown("""
            *   **Missing Value Imputation:**
                *   *Numerical:* Mean, Median.
                *   *Categorical:* Most Frequent (Mode).
            *   **Outlier Management:**
                *   IQR-based Clipping (Configurable).
                *   Applied **before** scaling.
            *   **Robustness:** Handles edge cases like empty data.
            """)
        with col_feat2:
            st.subheader("🔄 Feature Transformation")
            st.markdown("""
            *   **Categorical Encoding:**
                *   **One-Hot Encoding:** Handles unknown categories, cardinality limit.
                *   **Label Encoding:** Option for specific models.
            *   **Feature Scaling:**
                *   **Standard Scaling:** Z-score.
                *   **MinMax Scaling:** Configurable range.
            """)
        st.divider()
        st.subheader("⚙️ Pipeline & Usability")
        st.markdown("""
            *   **Model-Specific Pipelines:** Tailored defaults for Tree, Distance, Regression.
            *   **Stateful `fit`/`transform`:** Consistent processing for train/test data.
            *   **Detailed Summary Log:** Transparent step-by-step execution output.
            *   **UI Configuration:** Easy parameter selection.
            *   **Downloadable Results:** Processed data in CSV format.
        """)
    with tab_contact: # Variable name changed to match tab label
        st.markdown("<h3 style='text-align: center;'>Contact Us</h3>", unsafe_allow_html=True)
        st.write("") # Add some space

        # --- Formspree Contact Form (Keep as is) ---
        # !!! Remember to replace with YOUR actual Formspree endpoint !!!
        formspree_endpoint = "https://formspree.io/f/xdkeqegp" # Replace if needed

        st.markdown(f"""
        <form action="{formspree_endpoint}" method="POST" style="text-align: center;">
            <input type="text" name="name" placeholder="Your Name" required style="padding: 10px; margin: 5px; width: 80%; border: 1px solid #ccc; border-radius: 5px; max-width: 500px;">
            <br>
            <input type="email" name="_replyto" placeholder="Your Email" required style="padding: 10px; margin: 5px; width: 80%; border: 1px solid #ccc; border-radius: 5px; max-width: 500px;">
            <br>
            <input type="text" name="_subject" placeholder="Subject" style="padding: 10px; margin: 5px; width: 80%; border: 1px solid #ccc; border-radius: 5px; max-width: 500px;">
            <br>
            <textarea name="message" placeholder="Your Message" required style="padding: 10px; margin: 5px; width: 80%; height: 150px; border: 1px solid #ccc; border-radius: 5px; max-width: 500px;"></textarea>
            <br>
            <button type="submit" style="padding: 10px 20px; margin-top: 10px; background-color: #FF4B4B; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 1rem;">
                Send Message
            </button>
        </form>
        """, unsafe_allow_html=True)


else:
    selected_family = st.session_state.get('selected_family', None)
    st.header(f"{selected_family} Model Preprocessing Configuration")
    st.markdown(f"Configure the steps for the **{selected_family}** preprocessing pipeline.")

    # --- Dynamically Create Parameter Forms ---
    params = {}
    processor_class = None
    target_methods = [] # For model-specific fit methods if applicable

    # Prepare UI based on selected family
    if selected_family == "Tree-Based":
        processor_class = PreprocessTreeBased
        st.subheader("Common Options")
        params['numerical_imputation_strategy'] = st.selectbox("Numerical Imputation", ['median', 'mean'], key='tree_num_imp')
        params['categorical_imputation_strategy'] = st.selectbox("Categorical Imputation", ['most_frequent'], key='tree_cat_imp') # Add 'constant' later if needed
        params['outlier_handling_method'] = st.selectbox("Outlier Handling", ['iqr', None], format_func=lambda x: x if x else "None", key='tree_outlier')
        if params['outlier_handling_method'] == 'iqr':
            params['iqr_multiplier'] = st.number_input("IQR Multiplier", min_value=1.0, value=1.5, step=0.1, key='tree_iqr_mult')
        params['default_encoding_strategy'] = st.selectbox("Encoding Strategy", ['onehot', 'label'], help="Default encoding. Model-specific choices might override.", key='tree_enc')
        if params['default_encoding_strategy'] == 'onehot':
             params['max_onehot_features'] = st.number_input("Max Categories for OneHot", min_value=2, value=15, step=1, key='tree_max_ohe')
             params['handle_unknown_categories'] = st.selectbox("Handle Unknown Cats (OHE Transform)", ['ignore', 'error'], key='tree_unk_ohe')


        st.subheader("Target Model")
        st.markdown("Select the specific tree model you plan to use (this may adjust preprocessing steps).")
        # Define methods corresponding to the processor class
        target_methods = {
            "Auto (Defaults)": "fit", # Use the main fit method
            "Decision Tree / RF / ExtraTrees": "fit_for_decision_tree",
            "AdaBoost": "fit_for_adaboost",
            "Gradient Boosting (GBM)": "fit_for_gbm",
            "XGBoost": "fit_for_xgboost",
            "LightGBM (Native Categorical)": "fit_for_lightgbm",
            "CatBoost (Native Categorical)": "fit_for_catboost"
        }
        selected_method_name = st.selectbox("Select Target Model", list(target_methods.keys()), key='tree_model_select')
        actual_method_to_call = target_methods[selected_method_name]

        is_lgbm_native = "lightgbm" in actual_method_to_call
        is_catboost_native = "catboost" in actual_method_to_call
        
        if is_lgbm_native:
            st.checkbox("Impute Missing Before Native Handling?", value=True, key='tree_lgbm_impute', help="Should missing values be imputed even when using native handling?")
        # Add specific args for LightGBM/CatBoost native if needed
        if is_catboost_native:
            st.checkbox("Impute Missing Before Native Handling?", value=True, key='tree_catboost_impute', help="Should missing values be imputed even when using native handling?")


    elif selected_family == "Distance-Based":
        processor_class = PreprocessDistanceBased
        st.subheader("Configuration Options")
        params['numerical_imputation_strategy'] = st.selectbox("Numerical Imputation", ['median', 'mean'], key='dist_num_imp')
        params['categorical_imputation_strategy'] = st.selectbox("Categorical Imputation", ['most_frequent'], key='dist_cat_imp')
        params['encoding_strategy'] = st.selectbox("Encoding Strategy", ['onehot', 'label'], help="OneHot is strongly recommended for distance models.", key='dist_enc')
        if params['encoding_strategy'] == 'onehot':
             params['max_onehot_features'] = st.number_input("Max Categories for OneHot", min_value=2, value=15, step=1, key='dist_max_ohe')
             params['handle_unknown_categories'] = st.selectbox("Handle Unknown Cats (OHE Transform)", ['ignore', 'error'], key='dist_unk_ohe')
        params['outlier_handling_method'] = st.selectbox("Outlier Handling", ['iqr', None], format_func=lambda x: x if x else "None", key='dist_outlier')
        if params['outlier_handling_method'] == 'iqr':
            params['iqr_multiplier'] = st.number_input("IQR Multiplier", min_value=1.0, value=1.5, step=0.1, key='dist_iqr_mult')
        params['scaling_strategy'] = st.selectbox("Feature Scaling", ['standard', 'minmax', None], format_func=lambda x: x if x else "None", help="Scaling is CRUCIAL for distance models.", key='dist_scale')
        if params['scaling_strategy'] == 'minmax':
            range_options = {'(0, 1)': (0, 1), '(-1, 1)': (-1, 1)}
            selected_range_str = st.selectbox("MinMax Feature Range", list(range_options.keys()), key='dist_minmax_range')
            params['minmax_feature_range'] = range_options[selected_range_str]

        actual_method_to_call = "fit" # Distance uses the main fit method

    elif selected_family == "Regression-Based":
        processor_class = PreprocessRegressionBased
        st.subheader("Configuration Options")
        params['numerical_imputation_strategy'] = st.selectbox("Numerical Imputation", ['median', 'mean'], key='reg_num_imp')
        params['categorical_imputation_strategy'] = st.selectbox("Categorical Imputation", ['most_frequent'], key='reg_cat_imp')
        params['encoding_strategy'] = st.selectbox("Encoding Strategy", ['onehot', 'label'], help="OneHot is generally recommended for linear models.", key='reg_enc')
        if params['encoding_strategy'] == 'onehot':
             params['max_onehot_features'] = st.number_input("Max Categories for OneHot", min_value=2, value=15, step=1, key='reg_max_ohe')
             params['handle_unknown_categories'] = st.selectbox("Handle Unknown Cats (OHE Transform)", ['ignore', 'error'], key='reg_unk_ohe')
        params['outlier_handling_method'] = st.selectbox("Outlier Handling", ['iqr', None], format_func=lambda x: x if x else "None", key='reg_outlier')
        if params['outlier_handling_method'] == 'iqr':
            params['iqr_multiplier'] = st.number_input("IQR Multiplier", min_value=1.0, value=1.5, step=0.1, key='reg_iqr_mult')
        params['scaling_strategy'] = st.selectbox("Feature Scaling", ['standard', 'minmax', None], format_func=lambda x: x if x else "None", help="Scaling is recommended for most regression models.", key='reg_scale')
        if params['scaling_strategy'] == 'minmax':
            range_options = {'(0, 1)': (0, 1), '(-1, 1)': (-1, 1)}
            selected_range_str = st.selectbox("MinMax Feature Range", list(range_options.keys()), key='reg_minmax_range')
            params['minmax_feature_range'] = range_options[selected_range_str]

        actual_method_to_call = "fit" # Regression uses the main fit method

    # --- Process Button ---
    st.divider()
    process_button = st.button(f"Process Data for {selected_family}", type="primary")

    if process_button and processor_class is not None:
        current_df = st.session_state.original_df.copy()
        st.session_state.last_params = params.copy() # Store parameters used

        try:
            # Instantiate the correct processor with selected parameters
            processor = processor_class(**params)

            # Get the specific fit method to call (e.g., fit, fit_for_xgboost)
            fit_method = getattr(processor, actual_method_to_call)

            st.info(f"Running preprocessing using `{processor_class.__name__}.{actual_method_to_call}`...")
            with st.spinner("Processing... Please wait."):
                # Execute fit and capture output/summary
                # Special handling for methods that need specific args (like lightgbm native)
                if actual_method_to_call == "fit_for_lightgbm":
                     # Get outlier flag from common params
                     outlier_flag = params.get('outlier_handling_method') is not None
                     # Get impute flag directly from widget state
                     impute_flag = st.session_state.get('tree_lgbm_impute', True)

                     processed_df, summary = capture_processing_output(
                         fit_method,
                         current_df,
                         use_native_categorical=True, # Pass directly
                         impute_missing=impute_flag,  # Use widget value
                         handle_outliers=outlier_flag
                     )
                elif actual_method_to_call == "fit_for_catboost":
                     # Get outlier flag from common params
                     outlier_flag = params.get('outlier_handling_method') is not None
                     # Get impute flag directly from widget state
                     impute_flag = st.session_state.get('tree_catboost_impute', True)

                     processed_df, summary = capture_processing_output(
                         fit_method,
                         current_df,
                         use_native_categorical=True, # Pass directly
                         impute_missing=impute_flag,  # Use widget value
                         handle_outliers=outlier_flag
                     )
                # --- END MODIFIED SECTION ---
                elif hasattr(processor, actual_method_to_call) and actual_method_to_call != 'fit':
                    # Handle other specific fit methods like fit_for_xgboost etc.
                    fit_args = {}
                    # Dynamically check signature (safer than hardcoding varnames)
                    import inspect
                    sig = inspect.signature(fit_method)
                    if 'handle_outliers' in sig.parameters:
                         fit_args['handle_outliers'] = params.get('outlier_handling_method') is not None
                    if 'impute_missing' in sig.parameters:
                         fit_args['impute_missing'] = params.get('impute_missing', True) # May need adjustment based on model
                    if 'encoding_strategy' in sig.parameters:
                         fit_args['encoding_strategy'] = params.get('default_encoding_strategy', 'onehot')

                    processed_df, summary = capture_processing_output(fit_method, current_df, **fit_args)
                else:
                    # Default: call the main 'fit' method
                    processed_df, summary = capture_processing_output(fit_method, current_df)

            st.session_state.processed_df = processed_df
            st.session_state.fitted_processor = processor # Store the fitted processor
            st.session_state.processing_summary = summary # Store the captured output
            st.success("Preprocessing Complete!")

        except Exception as e:
            st.error(f"An error occurred during preprocessing:")
            st.exception(e) # Shows detailed traceback
            st.session_state.processed_df = None
            st.session_state.fitted_processor = None
            st.session_state.processing_summary = f"Error during processing: {e}"

    # --- Display Results and Summary ---
    if st.session_state.processed_df is not None:
        st.divider()
        st.subheader("📊 Processed Data")
        st.dataframe(st.session_state.processed_df)
        st.write(f"Shape after processing: {st.session_state.processed_df.shape}")

        # Download Button
        try:
            csv = st.session_state.processed_df.to_csv(index=False).encode('utf-8')
            st.download_button(
               label="Download Processed Data as CSV",
               data=csv,
               file_name=f'processed_{selected_family.lower().replace("-", "_")}.csv',
               mime='text/csv',
            )
        except Exception as e:
            st.warning(f"Could not prepare data for download: {e}")

        st.subheader("📝 Processing Summary")
        with st.expander("Show detailed steps and logs", expanded=False):
            st.code(st.session_state.processing_summary, language=None) # Display captured print output
            st.write("**Parameters Used:**")
            st.json(st.session_state.last_params) # Show parameters used for this run

    # --- FAQ Section (Example) ---
    st.divider()
    st.header("❓ Frequently Asked Questions")
    
    # --- Provided Example 1 ---
    with st.expander("What encoding strategy should I choose?"):
        st.markdown("""
            *   **One-Hot Encoding (OHE):** Generally recommended for **Linear Models (Regression-Based)** and **Distance-Based Models**. It avoids creating artificial order between categories. Can increase dimensionality significantly for high-cardinality features.
            *   **Label Encoding:** Mainly suitable for **Tree-Based Models** where the numerical split points handle the encoded values. Can also be used for LightGBM/CatBoost *native* categorical handling (requires specific parameters during model training). Avoid for linear/distance models unless categories have a true inherent order.
        """)
    
    # --- Provided Example 2 ---
    with st.expander("Why is scaling important?"):
        st.markdown("""
            Scaling (like Standard or MinMax) ensures that features with larger value ranges don't disproportionately influence models that rely on distances or magnitudes.
            *   **CRITICAL** for **Distance-Based Models** (KNN, SVM, Clustering) and **Regression Models** (Linear, Ridge, Lasso, SVR).
            *   **NOT needed** for **Tree-Based Models** (Decision Tree, Random Forest, GBM, XGBoost), as they split features one at a time irrespective of scale.
        """)
    
    # --- Rewritten FAQ 1 ---
    with st.expander("Are there any limitations to the tool?"):
        st.markdown("""
            Rudra's handy, but not limitless! Keep these in mind:
            *   **Big Data Blues:** Large files might hit memory limits (especially on free hosting).
            *   **Keep it Simple:** Super complex or niche preprocessing might need custom code.
            *   **Defaults ≠ Perfect:** Our settings are smart starting points, but your data might need extra TLC.
            *   **Cleaner, Not Creator:** We tidy up features, we don't invent new ones (that's feature engineering!).
        """)
    
    # --- Rewritten FAQ 2 ---
    with st.expander("What does \"Native Categorical Handling\" mean for LightGBM / CatBoost?"):
        st.markdown("""
            Think of it like a shortcut! These clever models can understand text categories *directly*, often skipping the need for manual One-Hot Encoding. Rudra sets things up so they can work their magic – potentially faster and sometimes even better!
        """)
    
    # --- Rewritten FAQ 3 ---
    with st.expander("What should I do if I encounter an error?"):
        st.markdown("""
            Uh oh! Before sounding the alarm, try these quick checks:
            *   **Clean CSV?:** Is your file format okay? No weird characters or messy rows?
            *   **Train/Test Twins?:** If using both, do they have the same columns?
            *   **Simplify:** Try turning off an option (like outlier handling) – did that fix it?
            *   **Still Stuck?:** The issue might be specific to your data. *(Consider adding: Reach out on GitHub [Link] if the problem persists!)*
        """)
