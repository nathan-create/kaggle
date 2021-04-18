import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('/home/runner/kaggle/titanic/train.csv')

keep_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
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

def convert_prediction_to_survival_val(entry):
    if entry < 0.5:
        return 0
    else:
        return 1

for var1 in features_to_use:
    for var2 in features_to_use[features_to_use.index(var1)+1:]:
        if not('Embarked=' in var1 and 'Embarked=' in var2):
            if not('CabinType=' in var1 and 'CabinType=' in var2):
                if not('SibSp' in var1 and 'SibSp' in var2):
                    columns_needed.append(var1 + " * " + var2)

interaction_features = columns_needed[1:]

for var in interaction_features:
    if ' * ' in var:
        vars = var.split(' * ')
        df[var] = df[vars[0]]*df[vars[1]]

def get_accuracy(predictions, actual):
    num_correct = 0
    num_incorrect = 0
    for i in range(len(predictions)):
        if predictions[i] == actual[i]:
            num_correct += 1
        else:
            num_incorrect += 1
    return num_correct/(num_correct + num_incorrect)

print("list of features:", interaction_features, '\n')

features_list = []
overall_testing_accuracy = 0
overall_training_accuracy = 0

while True:
    max_testing_accuracy = 0
    max_training_accuracy = 0
    new_feature = interaction_features[0]
    for current_feature in interaction_features:
        if current_feature not in features_list:
            training_df = df[:500]
            testing_df = df[500:]

            training_df = training_df[['Survived'] + features_list+[current_feature]]
            testing_df = testing_df[['Survived'] + features_list +[current_feature]]

            training_array = np.array(training_df)
            testing_array = np.array(testing_df)

            y_train = training_array[:,0]
            y_test = testing_array[:,0]

            X_train = training_array[:,1:]
            X_test = testing_array[:,1:]

            regressor = LogisticRegression(max_iter=1000)
            regressor.fit(X_train, y_train)

            y_test_predictions = regressor.predict(X_test)
            y_train_predictions = regressor.predict(X_train)

            y_test_predictions = [convert_prediction_to_survival_val(n) for n in y_test_predictions]
            y_train_predictions = [convert_prediction_to_survival_val(n) for n in y_train_predictions]

            training_accuracy = get_accuracy(y_train_predictions, y_train)
            testing_accuracy =  get_accuracy(y_test_predictions, y_test)

        if testing_accuracy > max_testing_accuracy:
            max_testing_accuracy = testing_accuracy
            max_training_accuracy = training_accuracy
            new_feature = current_feature
    
    if max_testing_accuracy <= overall_testing_accuracy:
        break
    
    max_overall_testing_accuracy = max_testing_accuracy
    max_overall_training_accuracy = max_training_accuracy
    features_list.append(new_feature)

    print("\n", features_list, "\n")
    print("training", max_overall_training_accuracy)
    print("testing", max_overall_testing_accuracy, "\n")
