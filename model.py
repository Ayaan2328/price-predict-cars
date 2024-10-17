import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
df = pd.read_csv('"C:\Users\Admin\Documents\CapstoneProject\price predictor\electric_cars.csv"')  # The dataset should have range, horsepower, and price

# Features and target
X = df[['range_km', 'horsepower']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
