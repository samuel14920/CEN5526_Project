import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app_data = pd.read_csv('app_data.csv')
mood_data = pd.read_csv('mood_data.csv')

# Merge the datasets on dat
# merged_data = pd.merge(app_data, mood_data, on=['Date', 'Time'], how='inner')
merged_data = pd.merge(app_data, mood_data, on=['Date'], how='inner')

# encode "App name" column as numeric values
merged_data['App name'] = pd.factorize(merged_data['App name'])[0]

# Split the data into training and testing sets
X = merged_data[['Duration_Seconds', 'App name']]
y = merged_data['Happiness']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing set
score = model.score(X_test, y_test)
print('R-squared:', score)

import matplotlib.pyplot as plt
plt.scatter(X['App name'], y)
plt.plot(X['App name'], model.predict(X), color='red')
plt.xlabel('App name')
plt.ylabel('Price')
plt.show()