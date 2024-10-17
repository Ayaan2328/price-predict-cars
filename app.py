from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the regression model
with open('model.pkl', 'rb') as file:
    model = joblib.load(file)

# Function to predict the price based on range and horsepower
def predict_price(range_value, horsepower):
    # Create a DataFrame for input
    input_data = pd.DataFrame({
        'Range (km)': [range_value],
        'Horsepower': [horsepower]
    })

    # Make a prediction
    predicted_price = model.predict(input_data)[0]
    return predicted_price

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        range_value = float(request.form['range'])
        horsepower = float(request.form['horsepower'])

        # Predict the price using the model
        predicted_price = predict_price(range_value, horsepower)

        # Format the predicted price
        if predicted_price >= 1_00_00_000:  # Greater than or equal to 1 crore
            formatted_price = f"{predicted_price / 1_00_00000:.2f} Cr"
        else:
            formatted_price = f"{predicted_price / 1_00_000:.2f} Lakhs"

        return render_template('result.html', formatted_price=formatted_price)
    except Exception as e:
        return str(e)  # Return the error message for debugging

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
