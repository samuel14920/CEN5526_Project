import math

import numpy
import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app_data = pd.read_csv('./Sam_participants/ParticipantFE/app_data.csv')
mood_data = pd.read_csv('./Sam_participants/ParticipantFE/mood_data.csv')
#place_data = pd.read_csv('./Sam_participants/ParticipantFE/place_data.csv')


# Merge the datasets on date and time
merged_data = pd.merge(app_data, mood_data, on=['Date'], how='inner')
print(merged_data)
# encode "App name" column as numeric values
Social_and_Media = ["Twitter", "Reddit", "Discord", "Instagram", "Facebook", "Telegram",
                    "Snapchat", "Spotify", "YouTube", "Gallery"]
Utility = ["Chrome", "Teams", "Gmail", "Call", "Drive", "Messages", "Maps", "Drive", "Weather", "Calculator",
           "Camera", "Contacts", "Google", "Clock", "Amazon Alexa"]
print(app_data)
print(app_data['Date'])
date_total = [[],[],[],[],[]]
for date in set(app_data['Date']):
    if type(date) == str:
        print(date)
        date_total[1].append(date)
        app_data_by_date = app_data.query("`Date` == @date & `App name` != 'Screen off (locked)'")
        mood_data_by_date = mood_data.query("`Date` == @date")
        social_app_by_date = app_data.query("`Date` == @date & `App name` in @Social_and_Media")
        utility_app_by_date = app_data.query("`Date` == @date & `App name` in @Utility")
        daily_total = app_data_by_date['Duration_Seconds'].sum()
        social_total = social_app_by_date['Duration_Seconds'].sum()
        utility_total = utility_app_by_date['Duration_Seconds'].sum()
        print(daily_total)
        daily_mood_avg = mood_data_by_date['Activity'].mean()
        print(daily_mood_avg)
        date_total[0].append(daily_mood_avg)
        date_total[2].append(daily_total)
        date_total[3].append(social_total)
        date_total[4].append(utility_total)
        print(app_data_by_date)
        print(daily_mood_avg)
        print(daily_total/3600)




print(date_total)

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
#totals = pd.DataFrame(date_total[1], columns=['Dates'])
#date_total[3], date_total[4]],
X_1 = pd.DataFrame(date_total[0], columns=['Mood'])
print(X_1)
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X_1, date_total[2], test_size=0.85, random_state=42)

# Fit a linear regression model
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

# Evaluate the model on the testing set
score = model.score(X_train, y_train)
coefficients = model.coef_
intercept = model.intercept_
print('R-squared:', score)
print('Coefficients:', coefficients )
print('Intercept:', intercept )