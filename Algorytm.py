import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
import lightgbm as lgb
from sklearn import metrics

data = pd.read_csv("Churn-Prediction-Dataset.csv")

data.drop(columns=["state", "area_code"],inplace=True)
data["international_plan"].replace({'yes' : 1, 'no' : 0}, inplace=True)
data["voice_mail_plan"].replace({'yes' : 1, 'no' : 0}, inplace=True)
data["churn"].replace({'yes' : 1, 'no' : 0}, inplace=True)

# print(data[:10])

# print(data.isnull().sum())

# sns.heatmap(data.corr())
# plt.show()

data.drop(columns=['total_day_minutes', 'total_eve_minutes', 'total_night_minutes', 'total_intl_minutes'],inplace=True)
y = data['churn']
X = data.drop("churn", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 64)

# clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
# models,predictions = clf.fit(X_train, X_test, y_train, y_test)
# models.to_csv("models.csv")
# predictions.to_csv("predictions.csv")

model = lgb.LGBMClassifier(verbose=0)
model.fit(X_train,y_train)

y_test_predicted = model.predict(X_test)

accuracy = metrics.accuracy_score(y_test,y_test_predicted)
confusion_matrix = metrics.confusion_matrix(y_test,y_test_predicted)

print('The accuracy score is ',accuracy*100,'%')
sns.heatmap(confusion_matrix)
plt.show()

model.booster_.save_model('customer_churn_predictor')