__author__ = 'Le Quang Hai'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
import statsmodels.api as st

#Loading data
train = pd.read_csv('train.csv', parse_dates=[0])
test = pd.read_csv('test.csv', parse_dates=[0])
print(train.head())

#Evaluation Function
for col in ['count', 'registered', 'casual']:
    train['log-' + col] = train[col].apply(lambda x: np.log1p(x))

#Date time separation
temp_train = pd.DatetimeIndex(train['datetime'])
train['year'] = temp_train.year
train['month'] = temp_train.month
train['hour'] = temp_train.hour
train['weekday'] = temp_train.weekday

temp_test = pd.DatetimeIndex(test['datetime'])
test['year'] = temp_test.year
test['month'] = temp_test.month
test['hour'] = temp_test.hour
test['weekday'] = temp_test.weekday
print(train['hour'].head())

#Categorize time to 3 section: 0-7, 8-15, 16-23
timeSection = []
timeSection_test = []
for item in train['hour']:
    if item<8:
        timeSection.append(1)
    elif item<16:
        timeSection.append(2)
    else:
        timeSection.append(3)
train['timeOfDay'] = np.log1p(timeSection)
for item in test['hour']:
    if item<8:
        timeSection_test.append(1)
    elif item<16:
        timeSection_test.append(2)
    else:
        timeSection_test.append(3)
test['timeOfDay'] = np.log1p(timeSection_test)

#Features vector
features = ['workingday', 'weather', 'temp', 'atemp', 'timeOfDay']

#Model building on the new training set
newtraining = train[:int(0.95*len(train))]
validation = train[int(0.95*len(train)):]

X = st.add_constant(newtraining[features])
model = st.OLS(newtraining['log-count'], X)
f = model.fit()
print(f.summary())

#Apply the model to test set
testnew = test[features]
testnew.insert(0, 'const', 1)
ypredtest = f.predict(testnew)
result = np.round(np.expm1(ypredtest))

df = pd.DataFrame({'datetime': test['datetime'], 'count': result})
df.to_csv('output.csv', index=False, columns=['datetime', 'count'])
