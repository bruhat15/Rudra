# rudra/tree/base.py

import pandas as pd
import numpy as np
# Use relative import to access utils from the common sibling directory
from ..common.utils import CommonPreprocessorUtils

class PreprocessTreeBased:
    """
    Orchestrates preprocessing specifically for Tree-Based models.

    Uses CommonPreprocessorUtils for underlying operations and manages the
    state of fitted transformers (imputers, encoders, outlier bounds).

    Provides a fit method to learn transformations and a transform method
    to apply them consistently. Offers convenience methods like
    'fit_for_xgboost' etc., which are essentially wrappers around 'fit'.
    """
    def __init__(self,
                 numerical_imputation_strategy='median',
                 categorical_imputation_strategy='most_frequent',
                 outlier_handling_method='iqr', # e.g., 'iqr' or None
                 iqr_multiplier=1.5,
                 default_encoding_strategy='onehot', # 'onehot' or 'label'
                 max_onehot_features=15, # Used if default_encoding_strategy is 'onehot'
                 handle_unknown_categories='ignore' # For OHE during transform
                ):
        """
        Initializes the orchestrator with configuration for tree-based models.

        Args:
            numerical_imputation_strategy (str): 'mean', 'median'.
            categorical_imputation_strategy (str): 'most_frequent'.
            outlier_handling_method (str | None): Method for outliers ('iqr' or None).
            iqr_multiplier (float): Multiplier for IQR if method is 'iqr'.
            default_encoding_strategy (str): Default for categorical features ('onehot', 'label').
            max_onehot_features (int): Max unique values for OHE before switching to Label Encoding.
            handle_unknown_categories (str): How OHE handles unknown cats during transform ('ignore' or 'error').
        """
        if default_encoding_strategy not in ['onehot', 'label', None]:
            raise ValueError("default_encoding_strategy must be 'onehot', 'label', or None")

        # --- Store Configuration ---
        self.config = {
            'num_impute_strategy': numerical_imputation_strategy,
            'cat_impute_strategy': categorical_imputation_strategy,
            'outlier_method': outlier_handling_method,
            'iqr_multiplier': iqr_multiplier,
            'default_encoding': default_encoding_strategy,
            'max_ohe_features': max_onehot_features,
            'ohe_handle_unknown': handle_unknown_categories,
        }

        # --- Initialize State Attributes (will be populated by fit) ---
        self._fitted_ = False # Flag to check if fit has been called
        self._common_utils = CommonPreprocessorUtils # Reference to utility class

        # Attributes to store fitted objects and info
        self._numerical_cols_original = None
        self._categorical_cols_original = None

        self._numerical_imputer = None
        self._categorical_imputers = {}
        self._outlier_bounds = {}
        self._label_encoders = {}       # Stores LE fitted objects {col_name: encoder}
        self._onehot_encoder = None         # Stores the single OHE fitted object
        self._onehot_encoded_cols_original = [] # Stores names of cols that went into OHE
        self._label_encoded_cols_original = []  # Stores names of cols that were Label Encoded

        # Store final columns after fit for potential validation during transform
        self._final_columns_after_fit = None

    def fit(self, df,
            impute_numerical=True,
            impute_categorical=True,
            handle_outliers=None, # If None, use config, else override
            encode_strategy=None, # If None, use config, else override
           ):
        """
        Fits the preprocessing steps on the input DataFrame based on configuration
        and method arguments.

        This method learns the necessary transformations (imputation values,
        outlier bounds, encodings) and stores them within the instance.
        It then applies these transformations to the input data.

        Args:
            df (pd.DataFrame): The training DataFrame to fit the preprocessor on.
            impute_numerical (bool): Override config: whether to impute numerical features.
            impute_categorical (bool): Override config: whether to impute categorical features.
            handle_outliers (bool | None): Override config: whether to handle outliers.
                                         If None, uses self.config['outlier_method'].
            encode_strategy (str | None): Override config: 'onehot', 'label', or None to skip encoding.
                                         If None, uses self.config['default_encoding'].

        Returns:
            pd.DataFrame: The transformed DataFrame after fitting and applying steps.
        """
        print("--- Starting PreprocessTreeBased Fit ---")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        df_processed = df.copy()

        # 1. Identify Feature Types on Original Data
        self._numerical_cols_original, self._categorical_cols_original = self._common_utils.identify_feature_types(df_processed)

        # --- Determine effective settings based on args and config ---
        effective_handle_outliers = self.config['outlier_method'] is not None if handle_outliers is None else handle_outliers
        effective_encode_strategy = self.config['default_encoding'] if encode_strategy is None else encode_strategy


        # --- Fit & Apply Imputation ---
        if impute_numerical:
            df_processed, self._numerical_imputer = self._common_utils.impute_numerical(
                df_processed, self._numerical_cols_original, self.config['num_impute_strategy']
            )
        if impute_categorical:
            df_processed, self._categorical_imputers = self._common_utils.impute_categorical(
                df_processed, self._categorical_cols_original, self.config['cat_impute_strategy']
            )

        # --- Fit & Apply Outlier Handling ---
        if effective_handle_outliers and self.config['outlier_method'] == 'iqr':
             # Apply only to original numerical columns *before* any potential encoding
            df_processed, self._outlier_bounds = self._common_utils.handle_outliers_iqr(
                df_processed, self._numerical_cols_original, self.config['iqr_multiplier']
            )
        else:
            print("Skipping outlier handling step.")
            self._outlier_bounds = {}

        # --- Fit & Apply Encoding ---
        self._label_encoders = {}
        self._onehot_encoder = None
        self._onehot_encoded_cols_original = []
        self._label_encoded_cols_original = []

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
                         # Check unique count on the potentially imputed data
                         unique_count = df_processed[col].nunique()
                         if unique_count <= self.config['max_ohe_features']:
                             cols_to_onehot.append(col)
                         else:
                             print(f"  - Column '{col}' has {unique_count} unique values (> {self.config['max_ohe_features']}), switching to Label Encoding.")
                             cols_to_label_instead.append(col)

                # Apply OHE where appropriate
                if cols_to_onehot:
                    df_processed, self._onehot_encoder, self._onehot_encoded_cols_original = self._common_utils.encode_onehot(
                        df_processed, cols_to_onehot, handle_unknown=self.config['ohe_handle_unknown']
                    )

                # Apply Label Encoding to the high-cardinality ones
                if cols_to_label_instead:
                    df_processed, label_encoders_high_card = self._common_utils.encode_label(
                        df_processed, cols_to_label_instead
                    )
                    self._label_encoders.update(label_encoders_high_card) # Add to overall label encoders
                    self._label_encoded_cols_original.extend(cols_to_label_instead)

            else:
                print(f"Warning: Encoding strategy '{effective_encode_strategy}' not recognized or applied.")
        elif not effective_encode_strategy:
            print("Skipping encoding step as per configuration/arguments.")
        else:
             print("No categorical columns found to encode.")


        # --- Finalization ---
        self._fitted_ = True
        self._final_columns_after_fit = df_processed.columns.tolist()
        print(f"--- PreprocessTreeBased Fit Completed. Final columns: {self._final_columns_after_fit} ---")
        return df_processed


    def transform(self, df):
        """
        Applies the fitted preprocessing steps to new data.

        Args:
            df (pd.DataFrame): New DataFrame to transform using the learned steps.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        print("--- Starting PreprocessTreeBased Transform ---")
        if not self._fitted_:
            raise RuntimeError("Preprocessor has not been fitted yet. Call the 'fit' method first.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        
        if df.empty:
            print("Input DataFrame is empty. Returning an empty DataFrame with columns learned during fit.")
            if self._final_columns_after_fit is not None:
                return pd.DataFrame(columns=self._final_columns_after_fit)
            else:
                # Fallback if fit wasn't fully successful or didn't produce columns
                print("Warning: Fit process did not define final columns. Returning original empty DataFrame.")
                return df.copy()

        df_processed = df.copy()

        # --- Apply Imputation (using stored imputers) ---
        df_processed = self._common_utils.transform_numerical_imputation(df_processed, self._numerical_cols_original, self._numerical_imputer)
        df_processed = self._common_utils.transform_categorical_imputation(df_processed, self._categorical_imputers)

        # --- Apply Outlier Handling (using stored bounds) ---
        df_processed = self._common_utils.transform_outlier_clipping(df_processed, self._outlier_bounds)

        # --- Apply Encoding (using stored encoders) ---
        # Apply label encoding first (on original columns designated for LE)
        if self._label_encoders:
            df_processed = self._common_utils.transform_label_encoding(df_processed, self._label_encoders)

        # Apply one-hot encoding (on original columns designated for OHE)
        if self._onehot_encoder:
            df_processed = self._common_utils.transform_onehot_encoding(df_processed, self._onehot_encoder, self._onehot_encoded_cols_original)

        # Optional: Reorder/select columns to match the output of fit if strictness is needed
        # try:
        #    df_processed = df_processed[self._final_columns_after_fit]
        #    print("Columns reordered to match fit output.")
        # except KeyError as e:
        #    print(f"Warning: Columns in transform differ from fit output. Missing columns: {e}. Check input data or handle_unknown settings.")
            # Decide how to handle missing columns - error, fill with default, etc.

        print("--- PreprocessTreeBased Transform Completed. ---")
        return df_processed

    # --- Convenience Fit Methods (Wrappers around fit) ---

    def fit_for_decision_tree(self, df, handle_outliers=True, encoding_strategy='onehot'):
        """Fits preprocessor for Decision Trees, RF, ExtraTrees."""
        print("\n--- Fitting Preprocessor for Decision Tree / Random Forest / ExtraTrees ---")
        # Standard requirements: Impute, Encode. Outliers recommended. Scale NOT needed.
        return self.fit(df,
                        impute_numerical=True,
                        impute_categorical=True,
                        handle_outliers=handle_outliers,
                        encode_strategy=encoding_strategy)

    def fit_for_gbm(self, df, handle_outliers=True, encoding_strategy='onehot'):
        """Fits preprocessor for standard Gradient Boosting Machines."""
        print("\n--- Fitting Preprocessor for Gradient Boosting Machine (GBM) ---")
        # Standard requirements: Impute, Encode. Outliers recommended. Scale NOT needed.
        return self.fit(df,
                        impute_numerical=True,
                        impute_categorical=True,
                        handle_outliers=handle_outliers,
                        encode_strategy=encoding_strategy)

    def fit_for_adaboost(self, df, handle_outliers=True, encoding_strategy='onehot'):
        """Fits preprocessor for AdaBoost."""
        print("\n--- Fitting Preprocessor for AdaBoost ---")
        # Standard requirements: Impute, Encode. Outliers *strongly* recommended. Scale NOT needed.
        return self.fit(df,
                        impute_numerical=True,
                        impute_categorical=True,
                        handle_outliers=handle_outliers, # Crucial for AdaBoost
                        encode_strategy=encoding_strategy)

    def fit_for_xgboost(self, df, handle_outliers=True, impute_missing=True, encoding_strategy='onehot'):
        """Fits preprocessor for XGBoost."""
        print("\n--- Fitting Preprocessor for XGBoost ---")
        # XGBoost can handle NaNs internally, but imputation often preferred. Requires encoding. Scale NOT needed.
        return self.fit(df,
                        impute_numerical=impute_missing,
                        impute_categorical=impute_missing,
                        handle_outliers=handle_outliers,
                        encode_strategy=encoding_strategy)

    def fit_for_lightgbm(self, df, handle_outliers=True, impute_missing=True, use_native_categorical=False):
        """Fits preprocessor for LightGBM."""
        print("\n--- Fitting Preprocessor for LightGBM ---")
        # Scale NOT needed.
        if use_native_categorical:
            print("Configuring for LightGBM Native Categorical Handling: Forcing Label Encoding.")
            # Requires imputation (or internal handling), Label Encoding for categoricals
            processed_df = self.fit(df,
                                    impute_numerical=impute_missing,
                                    impute_categorical=impute_missing,
                                    handle_outliers=handle_outliers,
                                    encode_strategy='label') # Force Label Encoding
            print("REMINDER: For LightGBM native handling, ensure categorical columns have 'category' dtype "
                  "or pass their names/indices via the 'categorical_feature' parameter during training.")
            print(f"(Label Encoded columns during fit: {self._label_encoded_cols_original})")
            return processed_df
        else:
            print("Configuring for LightGBM with standard encoding (using default or specified).")
            # Use standard pipeline (impute if desired, encode using default)
            return self.fit(df,
                            impute_numerical=impute_missing,
                            impute_categorical=impute_missing,
                            handle_outliers=handle_outliers,
                            encode_strategy=self.config['default_encoding']) # Use configured default

    def fit_for_catboost(self, df, handle_outliers=True, impute_missing=True, use_native_categorical=True):
        """Fits preprocessor for CatBoost."""
        print("\n--- Fitting Preprocessor for CatBoost ---")
        # Scale NOT needed.
        if use_native_categorical:
            print("Configuring for CatBoost Native Categorical Handling: Skipping explicit encoding.")
            # Only apply imputation and outlier handling if requested
            processed_df = self.fit(df,
                                    impute_numerical=impute_missing,
                                    impute_categorical=impute_missing, # Impute but don't encode
                                    handle_outliers=handle_outliers,
                                    encode_strategy=None) # <<< Explicitly skip encoding step

            print("REMINDER: For CatBoost native handling, pass the names or indices of the *original* "
                  f"categorical columns ({self._categorical_cols_original}) via the 'cat_features' parameter during training.")
            return processed_df
        else:
            print("Configuring for CatBoost with standard encoding (using default or specified).")
            # Use standard pipeline
            return self.fit(df,
                            impute_numerical=impute_missing,
                            impute_categorical=impute_missing,
                            handle_outliers=handle_outliers,
                            encode_strategy=self.config['default_encoding']) # Use configured default