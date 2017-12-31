import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

style.use('ggplot')

df = quandl.get("WIKI/GOOGL")
print("desc: ",df.describe())
print("axis: ",df.axes)

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
print("X length: ",len(X))

X = X[:-forecast_out]
print("X length: ",len(X))
print("x: ",X[:20])
print("--------------------------------------------------------------")
print("x_lately length: ",len(X_lately))
print("x_lately: ",X_lately[:20])


df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
# n_jobs ==> if is -1 then it will multithread as many jobs as processor can support
#
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

print("Head: \n",df.head())
print("Head: \n",df.count())

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# print("forecast set",forecast_set)

for i in forecast_set:
    # print("i=",i)
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    print("next_date: ",next_date)
    print("ith: ",[np.nan for _ in range(len(df.columns)-1)]+[i])
    # print("oth: ",[np.nan for _ in range(len(df.columns)-1)])
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

print("Head[after adjust]: ",df.head())


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()