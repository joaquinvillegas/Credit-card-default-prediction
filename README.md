# Credit Card Default Prediction

Predicting which credit card clients will have a payment delay of more than 30 days, and comparing explanatory and predictive models to detect them.

**Business goal:** help a lender prioritise manual reviews, request guarantees or ask for more information before granting credit.

*Notebook written in Spanish.*

## Data

[Credit Card Approval Prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) (Kaggle), two files:

- `application_record.csv`: 438,557 applications, 18 columns (client demographics and financial information).
- `credit_record.csv`: 1,048,575 monthly payment records.

The data is not included in this repository. Download both files from Kaggle and place them in a `data/` folder.

## Approach

1. **Target variable:** built `DEFAULT_EVER` (any payment more than 30 days late) from the monthly payment history and joined it to the applications by client → 25,134 clients, 12.3% defaults.
2. **Cleaning:** removed 134,203 rows with missing occupation (categorical, not imputable) and capped income and years employed with the IQR rule.
3. **Explanatory model:** logistic regression with Statsmodels.
4. **Predictive models:** logistic regression, Random Forest and neural network (scikit-learn).
5. **Class imbalance:** class weights and decision-threshold tuning to maximise default detection (recall).
6. **Overfitting control:** randomised hyperparameter search, stratified cross-validation, early stopping and L2 regularisation.

## Key results

| Model | Test AUC | Notes |
|---|---|---|
| **Random Forest** | **0.72** | Detects **79% of defaults** at a 0.35 threshold; 0.92 AUC on train shows some remaining overfitting |
| Neural network | 0.62 | Limited by dataset size and imbalance |
| Logistic regression | ≈ 0.54 | Barely above chance; does not capture non-linear relationships |

- **Precision is 17%:** out of 100 clients flagged, 17 actually default. Recall was prioritised because a missed default usually costs a lender more than reviewing a false alarm.
- **No model beats the majority rule on accuracy (87.7%):** the 36–54% accuracy is the cost of prioritising recall.
- **Explanatory logit pseudo R² = 0.005:** income and the other application variables explain very little of default behaviour.

## Limitations and next steps

- The application variables have weak predictive power; behavioural variables (payment history before the application, credit utilisation) would likely improve results.
- Random Forest still overfits; gradient boosting with stronger regularisation is a natural next step.

## How to run

Install the libraries: `pip install pandas numpy scikit-learn statsmodels matplotlib seaborn plotly kaleido jupyter`

Download `application_record.csv` and `credit_record.csv` from Kaggle, put them in a `data/` folder next to the notebook, and run `credit_card_default_prediction.ipynb`.

## Tools

Python · pandas · NumPy · scikit-learn · Statsmodels · Matplotlib · Seaborn · Plotly
