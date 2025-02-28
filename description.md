# Rudra Project Structure

## 📂 Project Overview
Rudra is a Python package focused on **data preprocessing** for machine learning. This document outlines the project structure, detailing the purpose of each file and folder to maintain a **well-organized and scalable codebase**.

---

## 📁 Project Structure
```
Rudra/
│
├── rudra/                     # Core Python package
│   ├── __init__.py            # Initializes package (add version here)
│   ├── preprocess.py          # Main logic (e.g., rd.preprocess())
│   ├── utils.py               # Helpers (data validation, logging, etc.)
│   ├── config.py              # Configuration (ML model mappings, NOT API keys)
│   └── exceptions.py          # Custom exceptions (optional but useful)
│
├── tests/                     # Unit/integration tests (mandatory for CI/CD)
│   ├── __init__.py
│   ├── test_preprocess.py     # Pytest tests for preprocessing
│   └── test_utils.py
│
├── examples/                  # Jupyter notebooks/tutorials
│   ├── basic_usage.ipynb
│   └── advanced_preprocessing.ipynb
│
├── docs/                      # Documentation (Sphinx/ReadTheDocs)
│   ├── source/
│   └── build/
│
├── .github/                   # GitHub Actions CI/CD workflows
│   └── workflows/
│       └── pytest.yml
│
├── setup.py                   # Package metadata (name, version, dependencies)
├── requirements.txt           # Dev + runtime dependencies
├── pyproject.toml             # Modern build system config (optional)
├── README.md                  # Project overview + installation guide
├── .gitignore                 # Exclude virtualenv, IDE files, etc.
└── LICENSE                    # Open-source license (MIT/Apache)
```

---

## 📌 Folder & File Breakdown

### **1️⃣ Core Package: `rudra/`**
Contains the main **preprocessing logic** and helper functions.

| File                 | Description |
|----------------------|-------------|
| `__init__.py`       | Initializes the package. Defines package-level imports and metadata (e.g., `__version__ = "0.1.0"`). |
| `preprocess.py`      | Implements **scaling & normalization** (MinMax, Standardization) and exposes them via `rd.preprocess()`. |
| `utils.py`          | Helper functions like **data validation, logging, and type checking** to keep `preprocess.py` clean. |
| `config.py`         | Stores **ML model mappings, parameter defaults, and configurations**. *(DO NOT store API keys!)* |
| `exceptions.py`     | Defines **custom exceptions** for error handling (e.g., `InvalidDataError`, `PreprocessingError`). *(Optional but useful!)* |

---

### **2️⃣ Testing: `tests/`**
Contains unit and integration tests to ensure the package works correctly.

| File                | Description |
|--------------------|-------------|
| `__init__.py`     | Marks the directory as a Python package. |
| `test_preprocess.py` | Tests all functions in `preprocess.py` using `pytest`. |
| `test_utils.py`   | Tests helper functions from `utils.py`. |

💡 **Why?**
- Prevents bugs and ensures correctness.
- Required for **CI/CD automation** (GitHub Actions).

---

### **3️⃣ Examples: `examples/`**
Contains Jupyter Notebooks demonstrating usage.

| File                     | Description |
|--------------------------|-------------|
| `basic_usage.ipynb`      | Simple examples of using `rudra.preprocess()`. |
| `advanced_preprocessing.ipynb` | Detailed use cases with real datasets. |

💡 **Why?**
- Helps users understand how to use the library.
- Useful for onboarding new contributors.

---

### **4️⃣ Documentation: `docs/`**
Contains project documentation.

| File/Folder       | Description |
|------------------|-------------|
| `source/`       | Raw `.rst` or `.md` files for documentation. |
| `build/`        | Compiled HTML/PDF documentation. |

💡 **Why?**
- Ensures the project is well-documented for users and contributors.
- Helps with **open-source contributions**.

---

### **5️⃣ GitHub CI/CD: `.github/`**
Contains GitHub Actions workflows for automated testing.

| File             | Description |
|----------------|-------------|
| `pytest.yml`   | Runs `pytest` on every push to validate code quality. |

💡 **Why?**
- Automates **testing & validation** before merging code.
- Prevents **broken code** from being deployed.

---

### **6️⃣ Build & Dependencies**
Defines package metadata and dependencies.

| File               | Description |
|------------------|-------------|
| `setup.py`       | Defines package details (name, version, dependencies). |
| `requirements.txt` | Lists **all dependencies** needed to run the project (`pandas`, `numpy`, etc.). |
| `pyproject.toml`  | Modern alternative to `setup.py`. *(Optional but recommended!)* |
| `README.md`       | Project overview, installation guide, and usage instructions. |
| `.gitignore`      | Prevents unnecessary files (e.g., `__pycache__/`, `.vscode/`, `.venv/`, `*.log`). |
| `LICENSE`         | Defines **open-source license** (e.g., MIT, Apache 2.0). |

---

## **🚀 Summary: Where to Put What?**

| **Feature**                  | **Where to Implement?** |
|------------------------------|-------------------------|
| Preprocessing Functions      | `rudra/preprocess.py` |
| Helper Functions (Validation, Logging) | `rudra/utils.py` |
| Default Parameters & Configurations | `rudra/config.py` |
| Custom Exception Handling    | `rudra/exceptions.py` |
| Unit Tests for Preprocessing | `tests/test_preprocess.py` |
| Usage Examples               | `examples/basic_usage.ipynb` |

---

## **💡 Next Steps**
Now that we have a clear structure:
✅ **Would you like me to write `preprocess.py` for you?**
✅ **Do you need test cases (`test_preprocess.py`)?**

