# 💳 Credit Scoring Model (Risk Analysis)

A machine learning project that predicts credit risk (Good/Bad) for borrowers. The script trains a base and an optimized Decision Tree model, comparing them using cross-validation and `GridSearchCV`.

**Dataset:** German Credit Data (1,000 customers × 9 features)

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree (Base) | ~0.71 | ~0.78 | ~0.84 | ~0.81 | ~0.65 |
| **Decision Tree (tuned)** | **~0.74** | **~0.76** | **~0.91** | **~0.83** | **~0.72** |

> *Exact values may vary slightly depending on the environment. Run the script to see precise results.*

### Key Findings
- **Checking account** status is the strongest predictor of credit risk.
- **Duration** and **Age** also play significant roles in determining if a loan is bad or good.
- The tuned model prioritizes avoiding false positives (giving a bad loan).

---

## 📁 Project Structure

```
Credit Scoring Model/
├── data/
│   ├── raw/
│   │   └── german_credit_data.csv   # Dataset
├── figures/                         # Generated performance charts
├── src/
│   └── cedit_scoring.py             # Complete ML pipeline
├── notebooks/                       # Exploratory analysis
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

### Setup & Run
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/credit-scoring-model.git
cd credit-scoring-model

# Create and activate virtual environment (optional)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the prediction script
python src/cedit_scoring.py
```

### Expected Output
The script will output data loading steps, feature extraction details, train/test shape, best parameters from GridSearchCV, a model comparison table, and a detailed feature importance breakdown. It will also generate several analytical `.png` graphs in the `figures/` directory.

---

## 🔧 What the Script Does

1. **Loads** the German Credit Scoring dataset
2. **Preprocesses** - handles missing values uniformly, encodes categorical targets and features, scales numerical data, and stratifies 70/30 splits.
3. **Trains Models:**
   - **Decision Tree (Base)**
   - **Decision Tree (Tuned)** - optimized hyperparameters via 5-Fold Stratified `GridSearchCV`.
4. **Evaluates** - logs Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
5. **Generates Visualizations** - Confusion Matrix, ROC Curve, and Feature Importances saved securely to the `figures/` directory.

---

## 🛠️ Technologies

- **Python 3**
- **pandas / NumPy** - Data manipulation
- **scikit-learn** - ML models, scaling, optimization
- **matplotlib / seaborn** - Data visualization

---

## 📝 License
This project is for educational purposes.