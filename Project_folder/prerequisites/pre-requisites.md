# Prerequisites

Before running the **Credit Card Approval Prediction** project, ensure that the following software and Python libraries are installed on your system.

## Software Requirements

| Software | Purpose | Download |
|----------|---------|----------|
| Python 3.10 or later | Programming language used for the project | https://www.python.org/downloads/ |
| Git | Clone and manage the project repository | https://git-scm.com/downloads |
| Anaconda Navigator (Optional) | Manage Python environments and packages | https://www.anaconda.com/download |
| Visual Studio Code / PyCharm (Optional) | IDE for developing and running the project | https://code.visualstudio.com/ / https://www.jetbrains.com/pycharm/ |

---

## Required Python Libraries

| Library | Purpose |
|---------|---------|
| NumPy | Numerical computing and array operations |
| Pandas | Data preprocessing and analysis |
| Scikit-learn | Machine learning algorithms and model evaluation |
| Matplotlib | Data visualization |
| Seaborn | Statistical data visualization |
| Flask | Web framework for deploying the ML model |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/jnanesh250/APSCHE-Credit_Card_Approval_prediction-.git
cd APSCHE-Credit_Card_Approval_prediction-
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### 3. Install the required dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the required libraries manually:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn flask
```

---

## Verify Installation

Run the following command to verify that all dependencies are installed successfully:

```bash
python -c "import numpy, pandas, sklearn, matplotlib, seaborn, flask; print('All dependencies installed successfully!')"
```

---

## Notes

- Python **3.10 or later** is recommended.
- Using a virtual environment is recommended to avoid dependency conflicts.
- Anaconda Navigator, VS Code, and PyCharm are optional development tools.