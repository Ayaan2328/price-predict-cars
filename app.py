from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the regression model
model = joblib.load('model.pkl')

# Function to make predictions based on user input
def predict_price(range_value, horsepower):
    # Create a DataFrame for the input features
    input_features = pd.DataFrame([[range_value, horsepower]], columns=['Range (km)', 'Horsepower'])
    
    # Make a prediction
    predicted_price = model.predict(input_features)
    return predicted_price[0]  # Return the predicted price

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input values from the form
        range_value = float(request.form['range'])
        horsepower = float(request.form['horsepower'])
        
        # Predict the price using the model
        predicted_price = predict_price(range_value, horsepower)

        return render_template('result.html', predicted_price=predicted_price)
    except Exception as e:
        return str(e)  # Return the error message for debugging

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
