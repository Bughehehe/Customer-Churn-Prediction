import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
# data.drop(columns=["churn"], inplace= True)
X = data.drop("churn", axis=1)
print(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 64)