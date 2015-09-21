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

#Drawing plots for train['workingday'] and train['season']
fig, axes = plt.subplots(nrows=2)

counts = collections.Counter(train['workingday'].values)
axes[0].bar(counts.keys(), counts.values(), color='red', align='center')
axes[0].set(title = 'Working Day')
axes[0].set_xticks(counts.keys())

counts = collections.Counter(train['season'].values)
axes[1].bar(counts.keys(), counts.values(), color='red', align='center')
axes[1].set(title = "Season")
axes[1].set_xticks(counts.keys())
plt.show()

#Drawing histogram of temp and atemp
fig, axes = plt.subplots(nrows=2)

axes[0].hist(train['temp'].values, color='red')
axes[0].set(title='Temp')
axes[1].hist(train['atemp'].values, color='red')
axes[1].set(title="Feels-like Temp")

plt.show()

#Drawing scattered matrix to discover relationship between variables
# sm = pd.scatter_matrix(train,alpha=0.5, figsize=(10,10), diagonal='hist')
# # === This code is needed to display all labels properly === [s.xaxis.label.set_rotation(45) for s in sm.reshape(-1)]
# [s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]
# # Offset label when rotating to prevent overlap of figure
# [s.get_yaxis().set_label_coords(-0.3,0.5) for s in sm.reshape(-1)]
# # Hide all ticks
# [s.set_xticks(()) for s in sm.reshape(-1)]
# [s.set_yticks(()) for s in sm.reshape(-1)]
#
# plt.show()

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

#Features vector
features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'year', 'month', 'weekday', 'hour']

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
df.to_csv('linearRegression_output.csv', index=False, columns=['datetime', 'count'])