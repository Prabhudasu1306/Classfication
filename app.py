from flask import Flask, request, render_template
import pandas as pd
from joblib import load

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model and polynomial transformer
model = load("Advertising_poly_model.joblib")
poly = load("poly_transformer.joblib")

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    tv_budget = float(request.form['tv'])
    radio_budget = float(request.form['radio'])
    newspaper_budget = float(request.form['newspaper'])

    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame([[tv_budget, radio_budget, newspaper_budget]], 
                               columns=["TV", "radio", "newspaper"])
    
    # Transform the input data using the polynomial transformer
    transformed_data = poly.transform(input_data)
    
    # Make prediction using the loaded model
    prediction = model.predict(transformed_data)[0]
    
    # Return the prediction result to the user
    return render_template('index.html', prediction_text=f"Predicted Sales: ${prediction:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
