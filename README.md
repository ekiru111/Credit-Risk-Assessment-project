# Credit Risk Assessment (Credit Default Risk Prediction)

## Project Title

Credit Default Risk Prediction

## Objective

Build and evaluate machine learning models to predict whether a customer will default on a loan, using demographic, financial, and credit history features.

## Dataset

- Source: `credit_default_dataset.csv`
- Rows: 1000
- Columns: 12 (including `CustomerID`, `Age`, `Gender`, `Income`, `CreditScore`, `LoanAmount`, `LoanTerm`, `NumOfCreditCards`, `NumOfLatePayments`, `HasDefaultedBefore`, `Default`, etc.)
- Missing values handled for: `Age`, `Income`, `CreditScore`, `LoanAmount`

## Data Preparation and Cleaning

1. Checked duplicates on `CustomerID` (no duplicates found).
2. Calculated missing value counts and percentages; imputed missing `Age`, `Income`, `CreditScore`, `LoanAmount` using median.
3. Generated new features:
   - `DebtToIncome` = `LoanAmount` / `Income`
   - `LoanPerCard` = `LoanAmount` / (`NumOfCreditCards` + 1)
   - `LatePaymentsPerMonth` = `NumOfLatePayments` / `LoanTerm`
   - `AgeGroup` = categorical bins (`Young`, `MidAge`, `Mature`, `Elderly`)
4. Dropped multicollinear features: `DebtToIncome`, `NumOfCreditCards`, `LatePaymentsPerMonth`.

## Exploratory Data Analysis (EDA)

- Univariate summaries for `Age`, `Gender`, `MaritalStatus`, `Education`, `Income`, `CreditScore`, `LoanAmount`.
- Visualizations with boxplots and pie charts for distributions.
- Correlation analysis and heatmap to inspect relationships and multicollinearity.

## Feature Engineering and Encoding

- Categorical values transformed via `LabelEncoder` (then `OneHotEncoder` in pipeline).
- Numeric features scaled using `StandardScaler`.

## Model Building

1. Train/test split: 80% train, 20% test, stratified by target (`Default`).
2. Models trained:
   - Logistic Regression (baseline + hyperparameter tuning with `GridSearchCV`)
   - Random Forest (baseline + hyperparameter tuning with `GridSearchCV`)
3. Preprocessing and model pipeline using `ColumnTransformer` and `Pipeline`.

## Evaluation Metrics

- Accuracy
- ROC AUC
- Classification report (precision, recall, f1-score)
- Confusion matrix

## Results

- Baseline Logistic Regression: ~69% accuracy, ~74% AUC.
- Tuned Logistic Regression: ~68.5% accuracy, ~75% AUC.
- Baseline Random Forest: ~67% accuracy, ~74% AUC.
- Tuned Random Forest: ~71% accuracy, ~77% AUC.

## Insights

- Random Forest with tuning was the best model in this exercise.
- Feature engineering on `LoanPerCard` and `AgeGroup` improved model signal.
- The dataset shows moderate separability and predictive risk factors in credit score and prior defaults.

## Next Steps

- Experiment with additional models: XGBoost, LightGBM.
- Apply further feature selection and class imbalance techniques if needed.
- Implement model persistence and a simple API for inference.
