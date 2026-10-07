# Customer Churn Prediction

A from-scratch machine-learning project that predicts whether a customer is likely to churn.

## Stack
Python, Pandas, scikit-learn, Joblib, Flask, HTML/CSS.

## Run
```bash
pip install -r requirements.txt
python train_model.py
python app.py
```
Then open the local Flask address.

## Data
The included CSV is a **synthetic demonstration dataset** generated for this portfolio project. It is not real customer data. Run `train_model.py` to train the model and print the test accuracy.

## Main workflow
1. Load data
2. Separate features and target
3. Scale numeric features
4. One-hot encode categorical features
5. Train Random Forest
6. Evaluate on a held-out test set
7. Serve predictions through Flask
