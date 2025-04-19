# rudra/distance/base.py

import pandas as pd
import numpy as np
# Use relative import to access utils from the common sibling directory
from ..common.utils import CommonPreprocessorUtils

class PreprocessDistanceBased:
    """
    Orchestrates preprocessing specifically for Distance-Based models (KNN, SVM, Clustering etc.).

    Uses CommonPreprocessorUtils for underlying operations and manages the
    state of fitted transformers (imputers, encoders, scalers, outlier bounds).

    Key steps typically include: Imputation, Outlier Handling, Categorical Encoding (OHE default),
    and Feature Scaling (Standard default).

    Provides a fit method to learn transformations and a transform method
    to apply them consistently.
    """
    def __init__(self,
                 numerical_imputation_strategy='median',
                 categorical_imputation_strategy='most_frequent',
                 encoding_strategy='onehot', # Default OHE for distance models
                 max_onehot_features=15,     # Used if encoding_strategy is 'onehot'
                 outlier_handling_method='iqr', # e.g., 'iqr' or None
                 iqr_multiplier=1.5,
                 scaling_strategy='standard', # 'standard', 'minmax', or None
                 minmax_feature_range=(0, 1), # Used if scaling_strategy is 'minmax'
                 handle_unknown_categories='ignore' # For OHE during transform
                ):
        """
        Initializes the orchestrator with configuration for distance-based models.

        Args:
            numerical_imputation_strategy (str): 'mean', 'median'.
            categorical_imputation_strategy (str): 'most_frequent'.
            encoding_strategy (str): 'onehot' or 'label'. OHE is generally preferred.
            max_onehot_features (int): Max unique values for OHE before switching to Label Encoding.
            outlier_handling_method (str | None): Method for outliers ('iqr' or None). Applied BEFORE scaling.
            iqr_multiplier (float): Multiplier for IQR if method is 'iqr'.
            scaling_strategy (str | None): 'standard', 'minmax', or None to skip scaling. CRUCIAL for distance models.
            minmax_feature_range (tuple): Range for MinMaxScaler, e.g., (0, 1) or (-1, 1).
            handle_unknown_categories (str): How OHE handles unknown cats during transform ('ignore' or 'error').
        """
        if encoding_strategy not in ['onehot', 'label', None]:
            raise ValueError("encoding_strategy must be 'onehot', 'label', or None")
        if scaling_strategy not in ['standard', 'minmax', None]:
             raise ValueError("scaling_strategy must be 'standard', 'minmax', or None")
        if encoding_strategy == 'label':
            print("Warning: Using 'label' encoding for distance-based models. "
                  "This implies ordinality which may not be appropriate. "
                  "'onehot' encoding is generally recommended.")
        if scaling_strategy is None:
             print("Warning: scaling_strategy is set to None. Scaling is CRITICAL for most distance-based models.")


        # --- Store Configuration ---
        self.config = {
            'num_impute_strategy': numerical_imputation_strategy,
            'cat_impute_strategy': categorical_imputation_strategy,
            'encoding_strategy': encoding_strategy,
            'max_ohe_features': max_onehot_features,
            'outlier_method': outlier_handling_method,
            'iqr_multiplier': iqr_multiplier,
            'scaling_strategy': scaling_strategy,
            'minmax_range': minmax_feature_range,
            'ohe_handle_unknown': handle_unknown_categories,
        }

        # --- Initialize State Attributes (will be populated by fit) ---
        self._fitted_ = False # Flag to check if fit has been called
        self._common_utils = CommonPreprocessorUtils # Reference to utility class

        # Attributes to store fitted objects and info
        self._numerical_cols_original = None
        self._categorical_cols_original = None
        self._cols_to_scale = [] # Store names of columns scaled during fit

        self._numerical_imputer = None
        self._categorical_imputers = {}
        self._outlier_bounds = {}
        self._label_encoders = {}
        self._onehot_encoder = None
        self._onehot_encoded_cols_original = []
        self._label_encoded_cols_original = []
        self._scaler = None # Will hold the fitted scaler (Standard or MinMax)

        # Store final columns after fit for potential validation during transform
        self._final_columns_after_fit = None

    def fit(self, df):
        """
        Fits the preprocessing steps on the input DataFrame based on configuration.

        Order of operations:
        1. Identify feature types.
        2. Impute missing values.
        3. Handle outliers (if configured).
        4. Encode categorical features.
        5. Scale numerical features (including newly encoded ones).

        Args:
            df (pd.DataFrame): The training DataFrame to fit the preprocessor on.

        Returns:
            pd.DataFrame: The transformed DataFrame after fitting and applying steps.
        """
        print("--- Starting PreprocessDistanceBased Fit ---")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if df.empty:
            print("Input DataFrame is empty. Skipping fitting process.")
            self._fitted_ = True # Mark as fitted, although no steps learned
            # Store original empty columns
            self._numerical_cols_original, self._categorical_cols_original = self._common_utils.identify_feature_types(df)
            self._final_columns_after_fit = df.columns.tolist()
            # Initialize state attributes to indicate nothing was fitted
            self._numerical_imputer = None
            self._categorical_imputers = {}
            self._outlier_bounds = {}
            self._label_encoders = {}
            self._onehot_encoder = None
            self._scaler = None
            self._cols_to_scale = []
            self._onehot_encoded_cols_original = []
            self._label_encoded_cols_original = []
            print(f"--- PreprocessDistanceBased Fit Completed (on empty DF). Final columns: {self._final_columns_after_fit} ---")
            return df.copy() # Return the empty DataFrame
        df_processed = df.copy()

        # 1. Identify Feature Types on Original Data
        self._numerical_cols_original, self._categorical_cols_original = self._common_utils.identify_feature_types(df_processed)

        # 2. Fit & Apply Imputation
        if self._numerical_cols_original:
             df_processed, self._numerical_imputer = self._common_utils.impute_numerical(
                df_processed, self._numerical_cols_original, self.config['num_impute_strategy']
            )
        if self._categorical_cols_original:
             df_processed, self._categorical_imputers = self._common_utils.impute_categorical(
                df_processed, self._categorical_cols_original, self.config['cat_impute_strategy']
            )

        # 3. Fit & Apply Outlier Handling (on original numerical columns BEFORE scaling)
        if self.config['outlier_method'] == 'iqr' and self._numerical_cols_original:
            df_processed, self._outlier_bounds = self._common_utils.handle_outliers_iqr(
                df_processed, self._numerical_cols_original, self.config['iqr_multiplier']
            )
        else:
            print("Skipping outlier handling step.")
            self._outlier_bounds = {}

        # 4. Fit & Apply Encoding
        self._label_encoders = {}
        self._onehot_encoder = None
        self._onehot_encoded_cols_original = []
        self._label_encoded_cols_original = []
        effective_encode_strategy = self.config['encoding_strategy']

        if self._categorical_cols_original and effective_encode_strategy:
            if effective_encode_strategy == 'label':
                print("Encoding Strategy: Label Encoding selected.")
                df_processed, self._label_encoders = self._common_utils.encode_label(
                    df_processed, self._categorical_cols_original
                )
                self._label_encoded_cols_original = list(self._label_encoders.keys())

            elif effective_encode_strategy == 'onehot':
                print("Encoding Strategy: OneHot Encoding selected (with Label Encoding fallback for high cardinality).")
                cols_to_onehot = []
                cols_to_label_instead = []
                for col in self._categorical_cols_original:
                     if col in df_processed.columns: # Check if column still exists
                         unique_count = df_processed[col].nunique()
                         if unique_count <= self.config['max_ohe_features']:
                             cols_to_onehot.append(col)
                         else:
                             print(f"  - Column '{col}' has {unique_count} unique values (> {self.config['max_ohe_features']}), switching to Label Encoding.")
                             cols_to_label_instead.append(col)

                if cols_to_onehot:
                    df_processed, self._onehot_encoder, self._onehot_encoded_cols_original = self._common_utils.encode_onehot(
                        df_processed, cols_to_onehot, handle_unknown=self.config['ohe_handle_unknown']
                    )

                if cols_to_label_instead:
                    df_processed, label_encoders_high_card = self._common_utils.encode_label(
                        df_processed, cols_to_label_instead
                    )
                    self._label_encoders.update(label_encoders_high_card)
                    self._label_encoded_cols_original.extend(cols_to_label_instead)

            else:
                 print(f"Warning: Encoding strategy '{effective_encode_strategy}' not recognized or applied.")
        else:
             print("Skipping encoding step (no categorical columns or strategy is None).")


        # 5. Fit & Apply Scaling (CRUCIAL STEP)
        # Identify all numerical columns *after* potential encoding
        current_numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
        self._cols_to_scale = current_numerical_cols # Store columns that will be scaled
        self._scaler = None

        if not self._cols_to_scale:
            print("No numerical columns found for scaling.")
        elif self.config['scaling_strategy'] == 'standard':
            print(f"Scaling Strategy: Standard Scaling selected for columns: {self._cols_to_scale}")
            df_processed, self._scaler = self._common_utils.scale_standard(
                df_processed, self._cols_to_scale
            )
        elif self.config['scaling_strategy'] == 'minmax':
             print(f"Scaling Strategy: MinMax Scaling (range={self.config['minmax_range']}) selected for columns: {self._cols_to_scale}")
             df_processed, self._scaler = self._common_utils.scale_minmax(
                df_processed, self._cols_to_scale, feature_range=self.config['minmax_range']
            )
        else:
            print("Skipping scaling step as per configuration.")


        # --- Finalization ---
        self._fitted_ = True
        self._final_columns_after_fit = df_processed.columns.tolist()
        print(f"--- PreprocessDistanceBased Fit Completed. Final columns: {self._final_columns_after_fit} ---")
        return df_processed


    def transform(self, df):
        """
        Applies the fitted preprocessing steps to new data in the correct order.

        Args:
            df (pd.DataFrame): New DataFrame to transform using the learned steps.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        print("--- Starting PreprocessDistanceBased Transform ---")
        if not self._fitted_:
            raise RuntimeError("Preprocessor has not been fitted yet. Call the 'fit' method first.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if df.empty:
            print("Input DataFrame for transform is empty. Returning empty DataFrame with columns from fit.")
            # Return an empty DataFrame but with the columns expected after processing
            if self._final_columns_after_fit is not None:
                return pd.DataFrame(columns=self._final_columns_after_fit)
            else:
                # Should not happen if fit was called, but as a fallback:
                return df.copy()
        
        df_processed = df.copy()

        # Apply steps in the same order as fit, using stored transformers

        # 1. Apply Imputation
        df_processed = self._common_utils.transform_numerical_imputation(df_processed, self._numerical_cols_original, self._numerical_imputer)
        df_processed = self._common_utils.transform_categorical_imputation(df_processed, self._categorical_imputers)

        # 2. Apply Outlier Handling
        df_processed = self._common_utils.transform_outlier_clipping(df_processed, self._outlier_bounds)

        # 3. Apply Encoding
        if self._label_encoders: # Apply label encoding first if it was used
            df_processed = self._common_utils.transform_label_encoding(df_processed, self._label_encoders)
        if self._onehot_encoder: # Apply OHE if it was used
            df_processed = self._common_utils.transform_onehot_encoding(df_processed, self._onehot_encoder, self._onehot_encoded_cols_original)

        # 4. Apply Scaling (using stored scaler and column list)
        if self._scaler and self._cols_to_scale:
             if self.config['scaling_strategy'] == 'standard':
                 df_processed = self._common_utils.transform_standard_scaling(df_processed, self._cols_to_scale, self._scaler)
             elif self.config['scaling_strategy'] == 'minmax':
                 df_processed = self._common_utils.transform_minmax_scaling(df_processed, self._cols_to_scale, self._scaler)
        elif self.config['scaling_strategy'] is not None:
            print("Warning: Scaling was configured during fit but is skipped during transform (no scaler or columns?).")

        # Optional: Reorder/select columns to match the output of fit
        # try:
        #    df_processed = df_processed[self._final_columns_after_fit]
        # except KeyError as e:
        #    print(f"Warning: Columns in transform differ from fit output. Missing columns: {e}. Check input data or handle_unknown settings.")

        print("--- PreprocessDistanceBased Transform Completed. ---")
        return df_processed

    # --- Convenience Fit Methods (Optional) ---
    # For distance-based models, the core pipeline is often sufficient.
    # We might add wrappers later if specific variations are common for KNN vs SVM etc.
    # Example:
    # def fit_for_knn(self, df, **kwargs):
    #     print("\n--- Fitting Preprocessor for KNN (Defaults) ---")
    #     # Could potentially override defaults, e.g., force standard scaling
    #     # self.config['scaling_strategy'] = 'standard' # Example override
    #     return self.fit(df)

    # def fit_for_svm(self, df, scale_range=(-1, 1), **kwargs):
    #      print("\n--- Fitting Preprocessor for SVM ---")
    #      # SVM sometimes benefits from [-1, 1] scaling
    #      self.config['scaling_strategy'] = 'minmax'
    #      self.config['minmax_range'] = scale_range
    #      print(f" Note: Forcing MinMax scaling with range {scale_range}")
    #      return self.fit(df)