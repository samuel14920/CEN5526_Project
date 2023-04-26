import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app_data = pd.read_csv('app_data.csv')
mood_data = pd.read_csv('mood_data.csv')


# Merge the datasets on date and time
merged_data = pd.merge(app_data, mood_data, on=['Date'], how='inner')
print(merged_data)
# encode "App name" column as numeric values

print(app_data)
print(app_data['Date'])
for date in set(app_data['Date']):
    print(date)
    app_data_by_date = app_data.query("`Date` == @date & `App name` != 'Screen off (locked)'")
    daily_total = app_data_by_date['Duration_Seconds'].sum()
    print(app_data_by_date)
    print(daily_total/3600)

# for app in app_data['App name']:
#     print(app)
#     name = app
#     app_data_by_name = app_data.query("`App name` == @name")

merged_data['App name'] = pd.factorize(merged_data['App name'])[0]
print(merged_data)
# Split the data into training and testing sets
X = merged_data[['Duration_Seconds', 'App name', 'Time_y']]
y = merged_data['Happiness']

# print(X)

# print("test")

# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing set
score = model.score(X_test, y_test)
print('R-squared:', score)