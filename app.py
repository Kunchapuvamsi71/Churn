from flask import Flask, render_template, request
import joblib, pandas as pd
app=Flask(__name__,template_folder="web")
model=joblib.load("models/churn_model.pkl")
@app.route("/",methods=["GET","POST"])
def home():
    result=None; probability=None
    if request.method=="POST":
        customer={"tenure_months":int(request.form["tenure_months"]),"monthly_charges":float(request.form["monthly_charges"]),"support_tickets":int(request.form["support_tickets"]),"contract":request.form["contract"],"internet_service":request.form["internet_service"],"payment_method":request.form["payment_method"],"senior_citizen":int(request.form["senior_citizen"]),"dependents":int(request.form["dependents"])}
        x=pd.DataFrame([customer]); pred=int(model.predict(x)[0]); probability=float(model.predict_proba(x)[0][1])
        result="Likely to churn" if pred else "Likely to stay"
    return render_template("index.html",result=result,probability=probability)
if __name__=="__main__": app.run(debug=True)
