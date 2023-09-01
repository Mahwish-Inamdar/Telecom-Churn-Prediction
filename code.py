import numpy as np
import pandas as pd
telcom = pd.read_csv("telco.csv", na_values = " ")
telcom.isnull().sum()
telcom.info()
telcom = telcom.fillna(0)
telcom.isnull().sum()
telcom = telcom.drop(["customerID"], axis = 1)
telcom.head()
column = telcom.columns.drop(['tenure', "SeniorCitizen", "MonthlyCharges", "TotalCharges"])
telcom_new = pd.get_dummies(telcom, columns = column , drop_first = True)
telcom_new.head()
telcom_new.groupby(["Churn_Yes"]).count()/telcom.shape[0]*100
target = telcom_new["Churn_Yes"]
variable = telcom_new.drop(["Churn_Yes"],axis =1)
variable.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(variable, target, test_size = 0.30, random_state = 42)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
knn = KNeighborsClassifier()
parameters = {"n_neighbors": np.arange(1, 100)}
knnclassifier = GridSearchCV(knn, parameters, cv = 10)
knnclassifier.fit(X_train, y_train)
print(knnclassifier.best_score_)
print(knnclassifier.best_params_)
rint(knnclassifier.cv_results_)
results = pd.DataFrame(knnclassifier.cv_results_)
results.head()
results.head()
import matplotlib.pyplot as plt
plt.plot(results["mean_test_score"])
knnclassifier.fit(X_test, y_test)
print(knnclassifier.best_score_)
print(knnclassifier.best_params_)
results = pd.DataFrame(knnclassifier.cv_results_)
plt.plot(results["mean_test_score"])
y_pred = knnclassifier.predict(X_test)
y_pred







