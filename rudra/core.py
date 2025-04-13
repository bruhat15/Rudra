# rudra/core.py

from rudra.executor import run_pipeline

def preprocess_data(df, remove_outliers=False, scale_data=False, encode_categoricals=False):
    """
    Wrapper to run preprocessing pipeline with user-defined options.
    
    Parameters:
        df (pd.DataFrame): The input dataset.
        remove_outliers (bool): Whether to remove outliers.
        scale_data (bool): Whether to scale numerical features.
        encode_categoricals (bool): Whether to encode categorical features.
    
    Returns:
        processed_df (pd.DataFrame): The preprocessed DataFrame.
        transformation_summary (List[str]): A summary of applied transformations.
    """
    # Pack options to pass along (you can expand this based on how your executor is structured)
    options = {
        "remove_outliers": remove_outliers,
        "scale_data": scale_data,
        "encode_categoricals": encode_categoricals,
    }

    # Run your internal pipeline (make sure executor supports these options or modify it to accept them)
    processed_df, summary = run_pipeline(df, **options)

    return processed_df, summary
