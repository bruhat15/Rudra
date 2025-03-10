# Rudra - One-Line ML Data Preprocessing

## 📌 Overview
Rudra is an open-source Python library that simplifies data preprocessing for Machine Learning.  
It allows users to transform datasets into ML-ready formats using a single function call.

## 🚀 Features
✔ One-line data preprocessing  
✔ Automated missing value handling  
✔ Categorical encoding (One-Hot, Label Encoding)  
✔ Scaling & Normalization  
✔ CSV-based `pandas.DataFrame` support  

## 📦 Dependencies
- Python >= 3.7
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- joblib >= 1.1.0

## 🔧 Installation
```bash
pip install rudra  # (In the future)
import rudra as rd
import pandas as pd
data=pd.read_csv(r"only csv text data")
rd.preprocess_MLregression(data)
rd.preprocess_MLdistance(data)
rd.preprocess_MLtree(data)
rd.preprocess_AI(data,"so the details about this data is extracted mean median mode of everything idk data.shape,data.describe and is forwarded along with this string that is telling what must be done is forwarded to gemini api, there gemini reads the data description and this message and generates preprocessing code which is brought back and executed on the local system on the data variable, note the user must configure a gemini api key in the library files for this to work else an error is returned with the instructions to do so")

the aim of making this project opensource is to let any individual or organisation improve on it and build any function as per their need