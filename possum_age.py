#%% Liblaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% Loading the data
data=pd.read_csv("possum.csv")
print(data.shape,data.dtypes,data.columns,data)

#%% Initial modifications
data.rename(columns={"Pop": "pop"},inplace=True)
data.drop("case",axis=1, inplace=True)
data['site'] = data['site'].apply(lambda x:str(x)) #site is cater categorical data

#%% Removing nullvalues
print(data.isna().sum())
data.dropna(inplace=True)
print(data.shape)


#%% Unique values
print(data.duplicated().sum()) #no duplicated rows
print(data.describe())
print(data.nunique())
for c in data.columns:
    print(f"Column {c} has {data[c].nunique()} unique values.")
    print(data[c].value_counts())

#%% Categorical data (site, poplation, sex) boxplots
fig, ax = plt.subplots(3, figsize=(10,15))
plt.suptitle("Categorical data \n Distribution of the age of possums", fontsize=20)
sns.boxplot(x='site', y="age", data=data, ax=ax[0])
sns.boxplot(x='pop', y="age", data=data, ax=ax[1])
sns.boxplot(x='sex', y="age", data=data, ax=ax[2])
plt.savefig("boxplots")
plt.show()

"""
There is a noticably higher variation in age amongst possums from Victoria
Median age of male possums is lesser than that of females and the variation is higher.
"""

#%% Numercial data - correlation
plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(),annot=True)
plt.savefig("correlation")
plt.show()

"""
The age of the possum has the highest correlation with the hdlngth, chest and belly size.
None of the features have significantly low correlation with age, we cannot discard any of them.
"""

#%% Splitting the data
from sklearn.model_selection import train_test_split
X = data.drop('age', axis = 1)
X = pd.get_dummies(X)
y = data['age']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=1)

#%% Linear Regression
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_predicted = linreg.predict(X_test)

# Accuracy 
from sklearn.metrics import mean_squared_error
print(f'RMSE:{np.around(np.sqrt(mean_squared_error(y_test, y_predicted)),3)}')
print(f'Standard Deviation of Age:{np.around(data.age.std(),3)}')
plt.scatter([i for i in range(21)],y_predicted, c="red", label="predicted")
plt.scatter([i for i in range(21)],y_test, c="green", label="actual")
plt.legend()
plt.show()

"""
We got Root Mean Square Error value: 1.759 with Standard Deviation of Age: 1.915
Let us try to use logarithmic function to get more normalized dataset in order to improve this model
"""

#%%  Getting the logharitm data
numerical = ['hdlngth','skullw','totlngth','taill','footlgth','earconch','eye','chest','belly']
categ = ['site_1', 'site_2', 'site_3', 'site_4', 'site_5','site_6', 'site_7', 'pop_Vic', 'pop_other', 'sex_f', 'sex_m']

X_train_log = X_train[numerical]
X_train_log=X_train_log.apply(lambda x: np.log(x+1))
X_train_log=X_train_log.join(X_train[categ])


X_test_log = X_test[numerical]
X_test_log=X_test_log.apply(lambda x: np.log(x+1))
X_test_log=X_test_log.join(X_test[categ])

y_train_log=y_train.apply(lambda x: np.log(x+1))


#%% Linear Regression for logharitms
linreglog =LinearRegression()
linreglog.fit(X_train_log, y_train_log)
y_predicted_log = linreglog.predict(X_test_log)

# Accuracy of logharitm model
y_predicted_log = np.exp(y_predicted_log)-1


print(f'RMSE value: {np.around(np.sqrt(mean_squared_error(y_test, y_predicted_log)),3)}')
print(f'Standard Deviation of Age: {np.around(data.age.std(),3)}')
plt.scatter([i for i in range(21)],y_predicted_log, label="predicted")
plt.scatter([i for i in range(21)],y_test, c="green", label="actual")
plt.legend()
plt.show()

"""
This time we have Root Mean Square Error value: 1.739 with Standard Deviation of Age: 1.915
There is a slight improvement.
"""
