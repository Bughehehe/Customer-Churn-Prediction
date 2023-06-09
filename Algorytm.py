import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
import lightgbm as lgb
from sklearn import metrics
import joblib

## Read data
data = pd.read_csv("Churn-Prediction-Dataset.csv")

## Preprocess data
data.drop(columns=["state", "area_code"],inplace=True)
data["international_plan"].replace({'yes' : 1, 'no' : 0}, inplace=True)
data["voice_mail_plan"].replace({'yes' : 1, 'no' : 0}, inplace=True)
data["churn"].replace({'yes' : 1, 'no' : 0}, inplace=True)

## Edition helpers
# print(data[:10])
# print(data.isnull().sum())
# sns.heatmap(data.corr())
# plt.show()

## Preprocess again
data.drop(columns=['total_day_minutes', 'total_eve_minutes', 'total_night_minutes', 'total_intl_minutes'],inplace=True)
y = data['churn']
X = data.drop(columns=["churn"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 64)

## Search for best model
# clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
# models,predictions = clf.fit(X_train, X_test, y_train, y_test)
# models.to_csv("models.csv")
# predictions.to_csv("predictions.csv")

## Create best model
model = lgb.LGBMClassifier(verbose=0)
model.fit(X_train,y_train)
y_test_predicted = model.predict(X_test)

# ## Print metrics
# accuracy = metrics.accuracy_score(y_test,y_test_predicted)
# confusion_matrix = metrics.confusion_matrix(y_test,y_test_predicted)
# print('The accuracy score is ',accuracy*100,'%')
# sns.heatmap(confusion_matrix)
# plt.show()

## Save best model
joblib.dump(model, 'model.txt')

## Read model
test_model = joblib.load('model.txt')

### Test model

test_data = pd.read_csv("test.csv")
check = pd.read_csv("sampleSubmission.csv")
test_data.drop(columns=["state", "area_code", 'id'],inplace=True)
test_data.drop(columns=['total_day_minutes', 'total_eve_minutes', 'total_night_minutes', 'total_intl_minutes'],inplace=True)
test_data["international_plan"].replace({'yes' : 1, 'no' : 0}, inplace=True)
test_data["voice_mail_plan"].replace({'yes' : 1, 'no' : 0}, inplace=True)
check["churn"].replace({'yes' : 1, 'no' : 0}, inplace=True)
test_y_test_predicted = test_model.predict(test_data)

print(test_y_test_predicted[:10])
