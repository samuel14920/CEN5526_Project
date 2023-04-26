import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Read the COVID-19 data CSV file
mood_data = pd.read_csv("mood_data.csv")

# Read the government responses CSV file
app_data = pd.read_csv("app_data.csv")

# Merge the two dataframes on the Country and Date columns
dataframe_merged = pd.merge(mood_data, app_data, on=["Date"], how="inner")

# Define the feature and target columns
features = ["App name", "Duration_Seconds"]
targets = ["Happiness", "Activity"]

print(dataframe_merged)

# Define the column transformer to preprocess the features
column_transformer = ColumnTransformer(
    [("ohe", OneHotEncoder(handle_unknown="ignore"), ["App name"])]
)

# Define the random forest regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Define the pipeline to preprocess the features and train the model
pipeline = Pipeline([
    ("preprocess", column_transformer),
    ("model", rf_model)
])

# Train the model for each target variable
for target in targets:
    # Split the data into training and testing sets
    X = dataframe_merged[features]
    y = dataframe_merged[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions on the testing set
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model performance using mean squared error and r-squared
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{target} Model:")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    
    # Plot the predicted vs actual values
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title(f"{target} Model: Predicted vs Actual Values")
    plt.show()
