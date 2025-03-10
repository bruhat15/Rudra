import pandas as pd
import google.generativeai as genai

# Replace with your actual API key
genai.configure(api_key="AIzaSyDKEBpe-jUkGjrqSpMyD81S1Wqi3WF9Xzg")

def preprocess_AI(df: pd.DataFrame, intent: str) -> pd.DataFrame:
    """
    Preprocesses a DataFrame based on a given intent using the Gemini API.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        intent (str): The user's intent.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """

    model = genai.GenerativeModel('gemini-pro')

    prompt = f"Given the following pandas dataframe and the intent: '{intent}', generate python code to complete the task. Do not include any explanation. Only include the code. Dataframe:\n{df.to_string()}"
    response = model.generate_content(prompt)
    generated_code = response.text

    try:
        local_vars = {"df": df}
        exec(generated_code, globals(), local_vars)
        df = local_vars["df"]
        return df
    except Exception as e:
        print(f"Error executing generated code: {e}\nGenerated Code:\n{generated_code}")  # Added context
        return df

if __name__ == "__main__":
    file_path = input("Enter the path to your CSV dataset: ")

    try:
        df_user = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        exit()
    except pd.errors.EmptyDataError:
        print(f"Error: Empty file at {file_path}")
        exit()
    except pd.errors.ParserError:
        print(f"Error: Could not parse file at {file_path}. Please ensure it is a valid CSV.")
        exit()

    intent = input("Enter your preprocessing intent: ")

    processed_df = preprocess_AI(df_user.copy(), intent)
    print("\nProcessed DataFrame:")
    print(processed_df)