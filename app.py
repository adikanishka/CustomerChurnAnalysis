import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the pre-trained model and dataset
model = pickle.load(open("churn_model.pkl", "rb"))
df_1 = pd.read_csv("tel_Churn.csv")

# Store the feature columns used during model training
with open('feature_columns.pkl', 'rb') as file:
    feature_columns = pickle.load(file)

@app.route("/")
def load_page():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    # Extract the form inputs
    input_data = [
        request.form['query1'], request.form['query2'], request.form['query3'], 
        request.form['query4'], request.form['query5'], request.form['query6'], 
        request.form['query7'], request.form['query8'], request.form['query9'], 
        request.form['query10'], request.form['query11'], request.form['query12'], 
        request.form['query13'], request.form['query14'], request.form['query15'], 
        request.form['query16'], request.form['query17'], request.form['query18'], 
        request.form['query19']
    ]

    # Create a DataFrame for the new input data
    input_df = pd.DataFrame([input_data], columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 
        'StreamingMovies', 'Contract', 'PaperlessBilling', 
        'PaymentMethod', 'tenure'
    ])
    
    # Process the 'tenure' column into bins
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    input_df['tenure_group'] = pd.cut(input_df.tenure.astype(int), bins=range(1, 80, 12), labels=labels, right=False)
    input_df.drop(columns=['tenure'], inplace=True)

    # Create dummy variables for categorical features
    input_dummies = pd.get_dummies(input_df, drop_first=True)

    # Align the input DataFrame with the model's expected feature names
    input_dummies = input_dummies.reindex(columns=model.feature_names_in_, fill_value=0)

    

    # Make a prediction using the pre-trained model
    prediction = model.predict(input_dummies)[0]
    probability = model.predict_proba(input_dummies)[0][1] * 100

    # Output the results
    if prediction == 1:
        result = "This customer is likely to churn!"
    else:
        result = "This customer is likely to stay."

    confidence = f"Confidence: {probability:.2f}% probability that he is in  the churn category."

    # Render the result on the home page
    return render_template('home.html', output1=result, output2=confidence, **request.form)

if __name__ == "__main__":
    app.run(debug=True)

