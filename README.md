# Heart-Disease-Prediction
Machine learning project that predicts whether a patient has heart disease based on medical measurements, comparing three different algorithms to find the most accurate one.

## What's in here
A single Jupyter notebook (`notebooks/heart_disease_analysis.ipynb`) that walks through:

- Loading and exploring the UCI Heart Disease dataset (Cleveland subset)
- Catching and fixing a duplicate-row issue that was inflating model scores
- Training three classifiers (logistic regression, decision tree, random forest) and comparing them with 5-fold cross-validation
- Tuning the best model with GridSearchCV
- Looking at which features mattered most

## Dataset
The dataset is the UCI Heart Disease dataset, downloaded from Kaggle. It has 1,025 rows and 13 features per patient - things like age, resting blood pressure, cholesterol, chest pain type, max heart rate during exercise and so on. The target is binary: 1 if the patient has heart disease, 0 otherwise. Classes are roughly balanced (~52% vs 48%).

One thing worth flagging: the raw file has 723 duplicate rows out of 1,025. I noticed because the decision tree initially scored 98.5% accuracy which felt too good. Dropping duplicates left 302 unique rows. After that, accuracy numbers came down to ~80% which is the honest performance.

## Results
| Model | 5-Fold CV Accuracy |
|---|---|
| Logistic Regression | 82.2% (±2.1%) |
| Decision Tree | 72.6% (±4.1%) |
| Random Forest | 81.3% (±3.0%) |
| Tuned Logistic Regression (L1, C=0.1) | 83.4% |

Logistic regression came out on top - both highest mean accuracy and lowest variance across folds. Decision tree looked competitive on a single train/test split but cross-validation showed it was actually weaker and less stable. After tuning with GridSearchCV (L1 penalty, C=0.1), the logistic regression model reached 83.4% CV accuracy.

For a medical screening context, recall on the disease class matters more than overall accuracy. The tuned model catches 85% of real heart disease cases at the cost of some false positives which is the right tradeoff when missing a real case is worse than an unnecessary follow-up test.

Most predictive features turned out to be chest pain type, number of major vessels visible on fluoroscopy and max heart rate during exercise - all of which line up with what cardiologists actually use.

## Stack
Python 3.11, pandas, numpy, scikit-learn, matplotlib, seaborn. Jupyter for the notebook.

## Running it yourself
1. Clone the repo
2. `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Download `heart.csv` from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) and put it in `data/`
4. Open the notebook and run all cells

## About
Built by Abrar Ali, I am 4th year CS student at Carleton University (AI/ML specialization).