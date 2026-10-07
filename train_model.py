import joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df=pd.read_csv("data/churn_data.csv")
X=df.drop(columns=["customer_id","churn"]); y=df["churn"]
num=["tenure_months","monthly_charges","support_tickets","senior_citizen","dependents"]
cat=["contract","internet_service","payment_method"]
pre=ColumnTransformer([("num",StandardScaler(),num),("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
model=Pipeline([("preprocessor",pre),("model",RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced"))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model.fit(Xtr,ytr)
print("Test accuracy:", model.score(Xte,yte))
joblib.dump(model,"models/churn_model.pkl")
print("Saved models/churn_model.pkl")
