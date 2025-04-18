import pandas as pd
import google.generativeai as genai
from typing import Optional
import os

def safe_preprocessing(df: pd.DataFrame, operation: str) -> pd.DataFrame:
    """
    Perform safe preprocessing without AI fallback
    """
    df = df.copy()
    
    operation = operation.lower()
    if "remove rows with null" in operation:
        return df.dropna()
    elif "remove columns with null" in operation:
        return df.dropna(axis=1)
    elif "fill numeric null" in operation:
        num_cols = df.select_dtypes(include='number').columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        return df
    elif "fill categorical null" in operation:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            df[col] = df[col].fillna('Unknown')
        return df
    else:
        print(f"Unknown operation: {operation}. Returning original DataFrame.")
        return df

def preprocess_with_ai(df: pd.DataFrame, intent: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """Optional AI-powered preprocessing"""
    if not api_key:
        return safe_preprocessing(df, intent)
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.0-pro')
        
        prompt = (
            f"Generate ONLY the pandas code to {intent} for this DataFrame:\n"
            f"{df.head().to_string()}\n\n"
            "Rules:\n"
            "1. Only use pandas (imported as pd)\n"
            "2. No explanations, comments, or markdown\n"
            "3. The variable must be named 'df'\n"
            "4. No external data or file operations\n"
            "Code:"
        )
        
        response = model.generate_content(prompt)
        code = response.text
        
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0]
        elif '```' in code:
            code = code.split('```')[1].split('```')[0]
        
        local_vars = {'df': df.copy(), 'pd': pd}
        exec(code, {}, local_vars)
        return local_vars.get('df', df)
        
    except Exception as e:
        print(f"AI preprocessing failed, using safe method instead. Error: {e}")
        return safe_preprocessing(df, intent)

def load_dataset():
    """Handle dataset loading from user"""
    while True:
        file_path = input("\nEnter path to your CSV file (or 'sample' for demo data): ").strip()
        
        if file_path.lower() == 'sample':
            data = {
                'Age': [25, None, 35, 45, 50],
                'Salary': [50000, 60000, None, 80000, 90000],
                'Department': ["HR", "Finance", None, "HR", "IT"],
                'Experience': [2, 5, None, 8, 10]
            }
            return pd.DataFrame(data)
        
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            continue
            
        try:
            return pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            print("Error: The file is empty")
        except pd.errors.ParserError:
            print("Error: Could not parse the file as CSV")
        except Exception as e:
            print(f"Error loading file: {e}")

def main():
    print("=== Data Preprocessor ===")
    print("You can process your own dataset or use our sample data\n")
    
    # Load dataset
    df = load_dataset()
    
    print("\nCurrent DataFrame (first 5 rows):")
    print(df.head())
    print(f"\nShape: {df.shape} rows, {df.shape[1]} columns")
    print("\nNull values per column:")
    print(df.isnull().sum())
    
    # User input
    intent = input("\nWhat preprocessing do you need? (e.g., 'remove rows with null values'): ")
    use_ai = input("Use AI-powered preprocessing? (y/n): ").lower() == 'y'
    
    api_key = None
    if use_ai:
        api_key = input("Enter your Gemini API key (or press Enter to use default): ").strip()
        if not api_key:
            # ⚠️ Default fallback API key (for testing only)
            api_key = "AIzaSyCcpI1-seLHcYTzC6po-IADi8ZR65Er99g"
    
    # Process data
    processed_df = preprocess_with_ai(df, intent, api_key)
    
    print("\nProcessed DataFrame (first 5 rows):")
    print(processed_df.head())
    print(f"\nNew shape: {processed_df.shape} rows, {processed_df.shape[1]} columns")
    
    # Save results
    if input("\nSave processed data? (y/n): ").lower() == 'y':
        while True:
            filename = input("Filename (default: processed.csv): ") or "processed.csv"
            try:
                processed_df.to_csv(filename, index=False)
                print(f"Successfully saved to {filename}")
                break
            except Exception as e:
                print(f"Error saving file: {e}. Please try another filename.")

if __name__ == "__main__":
    main()
