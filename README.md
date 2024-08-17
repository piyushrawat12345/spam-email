# spam-email
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv('C:\\Users\Piyush Rawat\spam mail.csv')
print(df)

data=df.where((pd.notnull(df)), ' ')
data.head(10)
data.info()
data.shape

data.loc[data['Category'] == 'spam', 'Category',] = 1
data.loc[data['Category'] == 'ham', 'Category',] = 0

x=data['Masseges']
y=data['Category']
print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
print(x.shape)
print(x_train.shape)
print(x_test.shape)
print(y.shape)
print(y_train.shape)
print(y_test.shape)

feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase='True')
fe = TfidfVectorizer()
x_train_features = fe.fit_transform(x_train)
x_test_features = fe.transform(x_test)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

print(x_train)
print(x_train_features)

model=LogisticRegression()
model.fit(x_train_features,y_train)
prediction_on_training_data = model.predict(x_train_features)
accuracy_on_training_data = accuracy_score(y_train , prediction_on_training_data)
print("Acc of training data: ",accuracy_on_training_data)
prediction_on_test_data = model.predict(x_test_features)
accuracy_on_test_data = accuracy_score(y_test , prediction_on_test_data)
print("acc of testing data: ",accuracy_on_test_data)

input_your_mail = ["let the music to be shown"]
input_data_features = fe.transform(input_your_mail)
prediction = model.predict(input_data_features)
print(prediction)
if prediction[0] == 1:
    print("Ham mail")
else:
    print("Spam mail")
