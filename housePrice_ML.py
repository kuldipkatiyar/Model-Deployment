import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
# Load your dataset file
from sklearn.datasets import fetch_california_housing
cali = fetch_california_housing(as_frame=True)
df=pd.concat([cali.data, cali.target], axis=1)

# Inspect data
df.head()
# Replace 'price' with the target column name
X = cali.data # Features
y = cali.target  # Target


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#4. Train the Model
# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

#5. Save the Model Using pickle
# Save the model to a file
with open('house_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)