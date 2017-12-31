import pandas as pd
import quandl as q
import numpy as np
import pandas as pd
import math
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression



df = q.get("WIKI/GOOGL")
df.to_csv(path_or_buf="./data.csv")
print(df.head())
print("------------start----------")
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

print(df.head())
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
print(df.head())
print("-----------------------------------------------------")

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
print("number of rows: \n",df.count())
forecast_out = int(math.ceil(0.01 * len(df)))
print("forecast value: ",forecast_out)


df['label'] = df[forecast_col].shift(-forecast_out)
print("result: ",df.tail(forecast_out+2))

df.dropna(inplace=True)
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
print("------------------------------------------")
print("X before scaling: ",X[:20])
X = preprocessing.scale(X)
print("-------------------------------------------")
print("X after scaling: ",X[:20])
print("-------------------------------------------")


y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# clf = svm.SVR()
# clf = svm.SVR(kernel='poly ')
for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print("SVM confidence: ",k,confidence)


cls  = LinearRegression()
cls.fit(X_train,y_train)
confidence_linear = cls.score(X_test,y_test)
print("Linear confidence: ",confidence_linear)



