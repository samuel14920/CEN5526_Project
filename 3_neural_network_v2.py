import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

app_data = pd.read_csv('app_data.csv')
mood_data = pd.read_csv('mood_data.csv')

# Merge the datasets on dat
# merged_data = pd.merge(app_data, mood_data, on=['Date', 'Time'], how='inner')
merged_data = pd.merge(app_data, mood_data, on=['Date'], how='inner')

# Preprocess data
merged_data = pd.get_dummies(merged_data, columns=['App name'])

# Split data into train and test sets
X = merged_data.drop(['Happiness', 'Activity', 'Date'], axis=1)
y = merged_data[['Happiness', 'Activity']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define neural network model
model = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', max_iter=100)

print(X_train)
print(y_train)

# Train model
model.fit(X_train, y_train)

# Evaluate model on test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: %.2f" % mse)

# Create residual plot
residuals = y_test - y_pred
sns.regplot(x=y_pred.values.reshape(-1), y=residuals.values.reshape(-1), lowess=True, color="g")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residual plot")
plt.show()

# Create scatter plot of actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Actual vs. predicted values")
plt.show()

# Calculate R-squared
r2 = model.score(X_test, y_test)
print(f"R-squared: {r2}")