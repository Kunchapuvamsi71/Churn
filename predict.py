import joblib, pandas as pd
model=joblib.load("models/churn_model.pkl")
customer={"tenure_months":8,"monthly_charges":95,"support_tickets":5,"contract":"Month-to-month","internet_service":"Fiber optic","payment_method":"Electronic check","senior_citizen":0,"dependents":0}
x=pd.DataFrame([customer]); pred=int(model.predict(x)[0]); prob=float(model.predict_proba(x)[0][1])
print(("Likely to churn" if pred else "Likely to stay"), f"- probability {prob:.1%}")
