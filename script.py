import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app_data = pd.read_csv('./FINAL_participant_data/1/app_data.csv')
mood_data = pd.read_csv('./FINAL_participant_data/1/mood_data.csv')

# # Normalize the time in app_data to match the times in mood_data
# app_data['Time'] = pd.to_datetime(app_data['Time'])
# app_data['Time'] = app_data['Time'].dt.floor('H')
# app_data['Time'] = app_data['Time'].dt.strftime('%I:%M%p').str.lower()

# Merge the datasets on date and time
# merged_data = pd.merge(app_data, mood_data, on=['Date', 'Time'], how='inner')
merged_data = pd.merge(app_data, mood_data, on=['Date'], how='inner')
print(merged_data)
# encode "App name" column as numeric values
merged_data['App name'] = pd.factorize(merged_data['App name'])[0]

print(merged_data)
# Split the data into training and testing sets
X = merged_data[['Duration_Seconds', 'App name', 'Time_y']]
y = merged_data['Happiness']

# print(X)

# print("test")

# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing set
score = model.score(X_test, y_test)
print('R-squared:', score)