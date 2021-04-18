import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import sys

df = pd.read_csv('/home/runner/kaggle/titanic/train.csv')

keep_cols = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']
df = df[keep_cols]

# sex
def convert_sex_to_int(sex):
    if sex == 'male':
        return 0
    elif sex == 'female':
        return 1
df['Sex'] = df['Sex'].apply(convert_sex_to_int)

#age
age_nan = df['Age'].apply(lambda entry: np.isnan(entry))
age_not_nan = df['Age'].apply(lambda entry: not np.isnan(entry))

mean_age = df['Age'][age_not_nan].mean()
df['Age'][age_nan] = mean_age

# sibSp
def indicator_greater_than_zero(x):
    if x > 0:
        return 1
    else:
        return 0
df['SibSp>0'] = df['SibSp'].apply(indicator_greater_than_zero)

# parch
df['Parch>0'] = df['Parch'].apply(indicator_greater_than_zero)
del df['Parch']

# cabinType
def get_cabin_type(cabin):
    if cabin != 'None':
        return cabin[0]
    else:
        return cabin

df['Cabin'] = df['Cabin'].fillna('None')
df['CabinType'] = df['Cabin'].apply(get_cabin_type)
for cabin_type in df['CabinType'].unique():
    name = 'CabinType='+cabin_type
    values = df['CabinType'].apply(lambda x: int(x==cabin_type))
    df[name] = values

del df['CabinType']
del df['Cabin']

# embarked
df['Embarked'] = df['Embarked'].fillna('None')
for embark in df['Embarked'].unique():
    name = 'Embarked='+embark
    values = df['Embarked'].apply(lambda x: int(x==embark))
    df[name] = values

del df['Embarked']

features_to_use = ['Sex','Pclass','Fare','Age','SibSp','SibSp>0','Parch>0','Embarked=C','Embarked=None','Embarked=Q','Embarked=S','CabinType=A','CabinType=B','CabinType=C','CabinType=D','CabinType=E','CabinType=F','CabinType=G','CabinType=None','CabinType=T']
columns_needed = ['Survived'] + features_to_use
df = df[columns_needed]

df_train = df[:500]
df_test = df[500:]
train_arr = np.array(df_train)
test_arr = np.array(df_test)

y_train = train_arr[:,0]
y_test = test_arr[:,0]
x_train = train_arr[:,1:]
x_test = test_arr[:,1:]

#linear regressor
regressor = LinearRegression()
regressor.fit(x_train, y_train)

def convert_prediction_to_survival_val(entry):
    if entry < 0.5:
        return 0
    else:
        return 1

y_test_predictions = regressor.predict(x_test)
y_test_predictions = [convert_prediction_to_survival_val(entry) for entry in y_test_predictions]
y_train_predictions = regressor.predict(x_train)
y_train_predictions = [convert_prediction_to_survival_val(entry) for entry in y_train_predictions]

def get_accuracy(predictions, actual):
    num_correct = 0
    num_incorrect = 0
    for i in range(len(predictions)):
        if predictions[i] == actual[i]:
            num_correct += 1
        else:
            num_incorrect += 1
    return num_correct/(num_correct + num_incorrect)

print('used features: ',features_to_use)
print('training accuracy: ',get_accuracy(y_train_predictions, y_train))
print('testing accuracy: ',get_accuracy(y_test_predictions, y_test))



#Logistic Regressor
regressor = LogisticRegression(max_iter=1000)
regressor.fit(x_train, y_train)

coefficients = {}
features_to_use = list(df_train.columns[1:])
feature_coefficients = list(regressor.coef_)[0]


for n in range(len(features_to_use)):
  column = features_to_use[n]
  coefficient = feature_coefficients[n]
  coefficients[column] = coefficient

y_test_predictions = regressor.predict(x_test)
y_train_predictions = regressor.predict(x_train)

y_test_predictions = [convert_prediction_to_survival_val(n) for n in y_test_predictions]
y_train_predictions = [convert_prediction_to_survival_val(n) for n in y_train_predictions]


print("\n", "features:", features_to_use, "\n")
print("training accuracy:", round(get_accuracy(y_train_predictions, y_train), 4))
print("testing accuracy:", round(get_accuracy(y_test_predictions, y_test), 4), "\n")

coefficients['constant'] = list(regressor.intercept_)[0]
print({k: round(v, 4) for k, v in coefficients.items()})

