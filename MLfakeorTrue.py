import numpy as np
import pandas as pd

data=pd.read_csv("train.csv")
data.info()

data=data.dropna()
data.isnull().sum()
data=data.drop(["id","author"],axis=1)
data

X=data.drop(["label"],axis=1)
Y=data["label"]
Y=list(Y.iloc[:2000])
x_title_list=list(X.iloc[:2000,0])
x_text_list=list(X.iloc[:2000,1])

len(x_title_list)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_title=cv.fit_transform(x_title_list).todense()

x_text=cv.fit_transform(x_text_list).todense()

x=np.hstack((x_title,x_text))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.25)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
