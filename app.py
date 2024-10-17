from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the regression model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    range_km = float(request.form['range'])
    horsepower = float(request.form['horsepower'])
    
    # Prepare input data for model
    features = [[range_km, horsepower]]
    
    # Make prediction
    predicted_price = model.predict(features)[0]
    
    return render_template('result.html', price=predicted_price)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
