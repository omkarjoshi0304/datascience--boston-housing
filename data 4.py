

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import mean_squared_error




column_names = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df=pd.read_csv('C:/Users/admin/Downloads/archive/HousingData.csv')
df['CHAS'].fillna(0,inplace=True)

for col in column_names:
    if df[col].count()!=506:
        df[col].fillna(df[col].median(),inplace=True)

x=df.drop('MEDV',axis=1)
y=df['MEDV']
# print(x.describe())

lr=LinearRegression()

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)



print(mean_squared_error(y_test,y_pred))