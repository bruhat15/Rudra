# rudra/common/utils.py
#this is a random comment to trigger a workflow for pipeline testing #2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings # To suppress potential warnings during transform if needed

class CommonPreprocessorUtils:
    """
    A utility class containing STATIC methods for common preprocessing tasks.
    These methods are generally stateless and operate directly on the input data,
    returning both the processed data and the fitted transformer object (if applicable).
    Includes corresponding 'transform_' methods to apply fitted transformers.
    """

    @staticmethod
    def identify_feature_types(df):
        """Identifies numerical and categorical columns."""
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"Identified numerical columns: {numerical_cols}")
        print(f"Identified categorical columns: {categorical_cols}")
        return numerical_cols, categorical_cols

    # --- Imputation ---

    # In rudra/common/utils.py

    # In rudra/common/utils.py

    @staticmethod
    def impute_numerical(df, numerical_cols, strategy='median'):
        """Fits a numerical imputer and transforms the data."""
        if not numerical_cols:
            print("No numerical columns specified for imputation.")
            return df.copy(), SimpleImputer(strategy=strategy) # Return unfitted

        df_processed = df.copy()
        cols_to_impute = [col for col in numerical_cols if col in df_processed.columns and df_processed[col].isnull().any()]

        if not cols_to_impute:
            print("No numerical imputation needed (no missing values found in specified columns or columns not found).")
            # Return an unfitted imputer if no actual imputation is done, but columns were specified
            return df_processed, SimpleImputer(strategy=strategy)

        imputer = SimpleImputer(strategy=strategy)
        print(f"Fitting numerical imputer (strategy='{strategy}') on columns: {cols_to_impute}")

        # Fit on columns needing imputation
        imputer.fit(df_processed[cols_to_impute])

        # Use get_feature_names_out to know which columns were actually processed (not skipped)
        try:
            # >= sklearn 1.0 ?
            features_out = imputer.get_feature_names_out(cols_to_impute)
        except AttributeError:
             # Fallback: Assume columns that were not all NaN were kept
             features_out = [col for col in cols_to_impute if not df_processed[col].isnull().all()]
             if len(features_out) != imputer.statistics_.shape[0]:
                 print("Warning: Could not reliably determine output features from imputer.")
                 features_out = cols_to_impute[:imputer.statistics_.shape[0]] # Best guess

                # Check if the features_out array is empty (meaning no features were processed)
        if features_out.size == 0: # <--- CORRECTED CHECK
            print("Warning: Numerical imputer did not process any features (check for all-NaN columns).")
            # Return original df, but with the fitted (potentially empty) imputer state
            return df_processed, imputer

        print(f"Applying numerical imputation transform using '{strategy}' to detected features: {features_out}")
        # Transform only the columns the imputer was fitted on
        transformed_data = imputer.transform(df_processed[cols_to_impute])

        # Create a new DataFrame with the transformed data and correct columns
        transformed_df = pd.DataFrame(transformed_data, columns=features_out, index=df_processed.index)

        # Update the original DataFrame, replacing only the columns that were transformed
        df_processed.update(transformed_df)

        print(f"Numerical imputation applied using '{strategy}'.")
        return df_processed, imputer

    @staticmethod
    def transform_numerical_imputation(df, numerical_cols, imputer):
        """Applies a fitted numerical imputer."""
        if not numerical_cols or not hasattr(imputer, 'transform') or not hasattr(imputer, 'statistics_'):
            # Check if imputer seems fitted (has statistics_)
            # print("Numerical imputation transform skipped (no columns, invalid imputer, or imputer not fitted).")
            return df.copy()

        df_processed = df.copy()
        cols_to_transform = [col for col in numerical_cols if col in df_processed.columns]
        if not cols_to_transform:
            return df_processed # No columns to transform

        print(f"Applying numerical imputation transform to columns: {cols_to_transform}")
        df_processed[cols_to_transform] = imputer.transform(df_processed[cols_to_transform])
        return df_processed

    @staticmethod
    def impute_categorical(df, categorical_cols, strategy='most_frequent'):
        """Fits categorical imputers and transforms the data."""
        if not categorical_cols:
            # print("No categorical columns to impute.")
            return df.copy(), {}

        df_processed = df.copy()
        imputers = {}
        imputed_cols_count = 0

        print(f"Fitting categorical imputer (strategy='{strategy}')")
        for col in categorical_cols:
            if col not in df_processed.columns:
                print(f"Warning: Categorical column '{col}' specified for imputation not found in DataFrame.")
                continue

            if df_processed[col].isnull().any():
                # Ensure column is object type for imputer
                df_processed[col] = df_processed[col].astype('object')
                imputer = SimpleImputer(strategy=strategy)
                # Reshape needed as SimpleImputer expects 2D array
                df_processed[col] = imputer.fit_transform(df_processed[[col]]).ravel()
                imputers[col] = imputer
                imputed_cols_count += 1
                print(f"  - Fitted imputer for column '{col}'.")

        if imputed_cols_count > 0:
            print(f"Categorical imputation applied to {imputed_cols_count} columns using '{strategy}'.")
        else:
            print("No missing values found in specified categorical columns for imputation.")

        return df_processed, imputers

    @staticmethod
    def transform_categorical_imputation(df, imputers):
        """Applies fitted categorical imputers."""
        if not imputers:
            # print("Categorical imputation transform skipped (no imputers provided).")
            return df.copy()

        df_processed = df.copy()
        print("Applying categorical imputation transform...")
        for col, imputer in imputers.items():
             if col in df_processed.columns and hasattr(imputer, 'transform') and hasattr(imputer, 'statistics_'):
                  df_processed[col] = df_processed[col].astype('object') # Ensure consistency
                  # Reshape needed as SimpleImputer expects 2D array
                  df_processed[col] = imputer.transform(df_processed[[col]]).ravel()
                  print(f"  - Applied transform for column '{col}'.")
             # else:
                  # Optional: Add warning if column is missing or imputer invalid
                  # print(f"  - Skipping transform for column '{col}' (not found or invalid imputer).")
        return df_processed

    # --- Outlier Handling ---

    @staticmethod
    def handle_outliers_iqr(df, numerical_cols, multiplier=1.5):
        """Calculates IQR bounds and clips outliers."""
        if not numerical_cols:
             # print("No numerical columns for IQR outlier handling.")
             return df.copy(), {}

        df_processed = df.copy()
        bounds = {}
        print("Calculating IQR bounds and clipping outliers...")
        for col in numerical_cols:
            if col not in df_processed.columns:
                 print(f"Warning: Column '{col}' specified for outlier handling not found.")
                 continue

            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1

            if pd.isna(Q1) or pd.isna(Q3):
                 print(f"  Skipping outlier handling for '{col}' (contains too many NaNs to calculate quantiles).")
                 continue

            if IQR == 0:
                lower_bound = Q1
                upper_bound = Q3
                print(f"  IQR is 0 for '{col}'. Using Q1/Q3 as bounds ({lower_bound:.2f}, {upper_bound:.2f}).")
            else:
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR

            bounds[col] = (lower_bound, upper_bound)

            initial_outliers = df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)].shape[0]
            if initial_outliers > 0:
                 print(f"  Clipping {initial_outliers} outliers in '{col}' outside range ({lower_bound:.2f}, {upper_bound:.2f}).")
                 df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
            else:
                 print(f"  No outliers detected in '{col}' based on IQR multiplier {multiplier}.")

        return df_processed, bounds

    @staticmethod
    def transform_outlier_clipping(df, bounds):
        """Applies learned outlier clipping bounds."""
        if not bounds:
            # print("Outlier clipping transform skipped (no bounds provided).")
            return df.copy()

        df_processed = df.copy()
        print("Applying outlier clipping transform...")
        for col, (lower, upper) in bounds.items():
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].clip(lower=lower, upper=upper)
                print(f"  - Applied clipping for column '{col}'.")
        return df_processed

    # --- Encoding ---

    @staticmethod
    def encode_label(df, categorical_cols):
        """Fits label encoders and transforms the data."""
        if not categorical_cols:
             # print("No categorical columns for Label Encoding.")
             return df.copy(), {}

        df_processed = df.copy()
        encoders = {}
        print(f"Fitting Label Encoder...")
        for col in categorical_cols:
            if col not in df_processed.columns:
                 print(f"Warning: Column '{col}' specified for label encoding not found.")
                 continue

            le = LabelEncoder()
            # Convert to string first to handle potential mixed types or NaNs if not imputed
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            encoders[col] = le
            print(f"  - Fitted and applied Label Encoder for column '{col}'.")
        return df_processed, encoders

    @staticmethod
    def transform_label_encoding(df, encoders):
        """Applies fitted label encoders."""
        if not encoders:
            # print("Label Encoding transform skipped (no encoders provided).")
            return df.copy()

        df_processed = df.copy()
        print("Applying Label Encoding transform...")
        with warnings.catch_warnings(): # Suppress warnings about unknown values for simplicity here
            warnings.simplefilter("ignore", category=UserWarning)
            for col, encoder in encoders.items():
                if col in df_processed.columns and hasattr(encoder, 'transform') and hasattr(encoder, 'classes_'):
                    df_processed[col] = df_processed[col].astype(str)
                    # Transform known values
                    known_mask = df_processed[col].isin(encoder.classes_)
                    if known_mask.any(): # Proceed only if there are known values
                         df_processed.loc[known_mask, col] = encoder.transform(df_processed.loc[known_mask, col])
                    # Handle unknowns: Convert column to numeric, forcing unknowns (non-transformed strings) to NaN, then fill with -1
                    # This assumes -1 is a suitable representation for unseen categories.
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(-1).astype(int)
                    print(f"  - Applied Label Encoding transform for column '{col}' (unknowns mapped to -1).")
        return df_processed

    @staticmethod
    def encode_onehot(df, categorical_cols, handle_unknown='ignore'):
        """Fits OneHotEncoder and transforms the data."""
        cols_to_encode = [col for col in categorical_cols if col in df.columns]
        if not cols_to_encode:
             print("No categorical columns found in DataFrame for One-Hot Encoding.")
             return df.copy(), None, []

        df_processed = df.copy()
        print(f"Fitting One-Hot Encoder (handle_unknown='{handle_unknown}') for columns: {cols_to_encode}")

        # Ensure columns are string type before OHE
        for col in cols_to_encode:
            df_processed[col] = df_processed[col].astype(str)

        encoder = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown)
        encoded_data = encoder.fit_transform(df_processed[cols_to_encode])

        # Create new DataFrame with encoded columns
        try:
            # get_feature_names_out is preferred
             encoded_feature_names = encoder.get_feature_names_out(cols_to_encode)
        except AttributeError:
             # Fallback for older scikit-learn versions
             encoded_feature_names = [f"{col}_{cat}" for i, col in enumerate(cols_to_encode) for cat in encoder.categories_[i]]


        encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=df_processed.index)

        # Drop original categorical columns and join encoded ones
        df_processed = df_processed.drop(columns=cols_to_encode)
        df_processed = pd.concat([df_processed, encoded_df], axis=1)
        print(f"One-Hot Encoding applied. New columns: {list(encoded_feature_names)}")

        return df_processed, encoder, cols_to_encode # Return original names that were encoded


    @staticmethod
    def transform_onehot_encoding(df, encoder, original_ohe_cols):
        """Applies a fitted one-hot encoder. Handles missing original columns."""
        if not encoder or not original_ohe_cols or not hasattr(encoder, 'transform'):
            # print("One-Hot Encoding transform skipped (no encoder, no original columns specified, or invalid encoder).")
            return df.copy()

        df_processed = df.copy()
        if df_processed.empty:
            print("Skipping One-Hot Encoding transform (input DataFrame is empty).")
            # Ensure required columns exist if we didn't return early from orchestrator
            # This part needs care: If OHE adds columns, the empty DF needs them.
            # The check in the orchestrator is likely better for handling this.
            # For simplicity here, just return the empty df.
            # A more robust util might add the expected OHE columns with 0 rows.
            return df_processed # Return empty df

        print(f"Applying One-Hot Encoding transform based on original columns: {original_ohe_cols}...")

        # Identify which of the original OHE columns ACTUALLY exist in the current df
        cols_present = [col for col in original_ohe_cols if col in df_processed.columns]
        cols_missing = [col for col in original_ohe_cols if col not in df_processed.columns]

        if cols_missing:
            print(f"  - Original OHE columns missing in input data: {cols_missing}. Will generate 0s for their features.")

        # Prepare data for transform: Use present columns, add missing ones with a placeholder (like NaN)
        # Ensure columns are in the same order as during fit
        data_for_transform = pd.DataFrame(index=df_processed.index)
        for col in original_ohe_cols:
             if col in cols_present:
                 # Convert to string to match fit-time expectation and handle NaNs consistently
                 data_for_transform[col] = df_processed[col].astype(str)
             else:
                 # Add missing column with NaNs (or another placeholder compatible with handle_unknown='ignore')
                 # SimpleImputer within OHE pipeline handles this better, but for standalone OHE:
                 # Using NaN string representation often works with handle_unknown='ignore'
                 data_for_transform[col] = pd.Series([np.nan] * len(df_processed), index=df_processed.index).astype(str)

        # Perform the transformation using the prepared data
        encoded_data = encoder.transform(data_for_transform[original_ohe_cols]) # Ensure order

        # Get feature names using the original column list
        try:
             encoded_feature_names = encoder.get_feature_names_out(original_ohe_cols)
        except AttributeError:
             # Fallback for older versions (less reliable if categories changed subtly)
             encoded_feature_names = [f"{col}_{cat}" for i, col in enumerate(original_ohe_cols) for cat in encoder.categories_[i]]

        encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=df_processed.index)

        # Drop original columns IF they were present in the input df
        df_processed = df_processed.drop(columns=cols_present, errors='ignore')

        # Concatenate: Join the processed df (without original OHE cols if they existed)
        # with the newly created OHE columns
        df_processed = pd.concat([df_processed, encoded_df], axis=1)

        print(f"  - One-Hot Encoding transform applied. Added/updated columns: {list(encoded_feature_names)}")
        return df_processed

    # --- Scaling (Added for other modules) ---

    @staticmethod
    def scale_standard(df, numerical_cols):
        """Fits StandardScaler and transforms the data."""
        cols_to_scale = [col for col in numerical_cols if col in df.columns]
        if df.empty or not cols_to_scale:
            print("Skipping Standard Scaling (DataFrame is empty or no columns to scale).")
            return df.copy(), StandardScaler()

        df_processed = df.copy()
        scaler = StandardScaler()
        print(f"Fitting StandardScaler on columns: {cols_to_scale}")
        df_processed[cols_to_scale] = scaler.fit_transform(df_processed[cols_to_scale])
        print("Standard Scaling applied.")
        return df_processed, scaler

    @staticmethod
    def transform_standard_scaling(df, numerical_cols, scaler):
        """Applies a fitted StandardScaler."""
        cols_to_transform = [col for col in numerical_cols if col in df.columns]
        if not scaler or not cols_to_transform or not hasattr(scaler, 'transform') or not hasattr(scaler, 'mean_'): # Check if fitted
            # print("Standard Scaling transform skipped.")
            return df.copy()

        df_processed = df.copy()
        if df_processed.empty:
            print("Skipping Standard Scaling transform (input DataFrame is empty).")
            return df_processed # Return empty df

        print(f"Applying Standard Scaling transform to columns: {cols_to_transform}")
        df_processed[cols_to_transform] = scaler.transform(df_processed[cols_to_transform])
        return df_processed

    @staticmethod
    def scale_minmax(df, numerical_cols, feature_range=(0, 1)):
        """Fits MinMaxScaler and transforms the data."""
        cols_to_scale = [col for col in numerical_cols if col in df.columns]
        if df.empty or not cols_to_scale:
            print("Skipping MinMax Scaling (DataFrame is empty or no columns to scale).")
            return df.copy(), MinMaxScaler(feature_range=feature_range) # Return unfitted scaler

        df_processed = df.copy()
        scaler = MinMaxScaler(feature_range=feature_range)
        print(f"Fitting MinMaxScaler (range={feature_range}) on columns: {cols_to_scale}")
        df_processed[cols_to_scale] = scaler.fit_transform(df_processed[cols_to_scale])
        print("MinMax Scaling applied.")
        return df_processed, scaler

    @staticmethod
    def transform_minmax_scaling(df, numerical_cols, scaler):
        """Applies a fitted MinMaxScaler."""
        cols_to_transform = [col for col in numerical_cols if col in df.columns]
        if not scaler or not cols_to_transform or not hasattr(scaler, 'transform') or not hasattr(scaler, 'min_'): # Check if fitted
             # print("MinMax Scaling transform skipped.")
             return df.copy()

        df_processed = df.copy()
        print(f"Applying MinMax Scaling transform to columns: {cols_to_transform}")
        df_processed[cols_to_transform] = scaler.transform(df_processed[cols_to_transform])
        return df_processed