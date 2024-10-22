import numpy as np
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

boston_df = pd.read_csv("boston_housing.csv")
org_df = pd.read_csv("boston_housing.csv")

boston_df = boston_df.drop(boston_df.columns[0], axis=1)
org_df = org_df.drop(org_df.columns[0], axis=1)

boston_df

boston_df.info()

boston_df.describe()

boston_df.columns

boston_df.hist(bins=50, figsize=(20,10))
plt.suptitle('Feature Distribution', x=0.5, y=1.02, ha='center', fontsize='large')
plt.tight_layout()
plt.show()

plt.figure(figsize=(20,20))
plt.suptitle('Pairplots of features', x=0.5, y=1.02, ha='center', fontsize='large')
sns.pairplot(boston_df.sample(250))
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x=boston_df.MEDV)
plt.title("Boxplot for MEDV")
plt.show()

plt.figure(figsize=(10,5))
sns.distplot(a=boston_df.CHAS,bins=10, kde=False)
plt.title("Histogram for Charles river")
plt.show()

boston_df.loc[(boston_df["AGE"] <= 35),'group_age'] = '35 years and younger'
boston_df.loc[(boston_df["AGE"] > 35) & (boston_df["AGE"]<70),'group_age'] = 'between 35 and 70 years'
boston_df.loc[(boston_df["AGE"] >= 70),'group_age'] = '70 years and older'

boston_df

plt.figure(figsize=(10,5))
sns.boxplot(x=boston_df.MEDV, y=boston_df.group_age, data=boston_df)
plt.title("Boxplot for the MEDV variable vs the AGE variable")
plt.show()

plt.figure(figsize=(10,5))
sns.scatterplot(x=boston_df.NOX, y=boston_df.INDUS, data=boston_df)
plt.title("Relationship between NOX and INDUS")
plt.show()

plt.figure(figsize=(10,5))
sns.distplot(a=boston_df.PTRATIO,bins=10, kde=False)
plt.title("Histogram for the pupil to teacher ratio variable")
plt.show()

boston_df

boston_df["CHAS"].value_counts()

a = boston_df[boston_df["CHAS"] == 0]["MEDV"]
a

b = boston_df[boston_df["CHAS"] == 1]["MEDV"]
b

scipy.stats.ttest_ind(a,b,axis=0,equal_var=True)

boston_df["AGE"].value_counts()

boston_df.loc[(boston_df["AGE"] <= 35),'group_age'] = '35 years and younger'
boston_df.loc[(boston_df["AGE"] > 35) & (boston_df["AGE"]<70),'group_age'] = 'between 35 and 70 years'
boston_df.loc[(boston_df["AGE"] >= 70),'group_age'] = '70 years and older'

boston_df

low = boston_df[boston_df["group_age"] == '35 years and younger']["MEDV"]
mid = boston_df[boston_df["group_age"] == 'between 35 and 70 years']["MEDV"]
high = boston_df[boston_df["group_age"] == '70 years and older']["MEDV"]

f_stats, p_value = scipy.stats.f_oneway(low,mid,high,axis=0)

print("F-Statistic={0}, P-value={1}".format(f_stats,p_value))

pearson,p_value = scipy.stats.pearsonr(boston_df["NOX"],boston_df["INDUS"])

print("Pearson Coefficient value={0}, P-value={1}".format(pearson,p_value))

boston_df.columns

y = boston_df['MEDV']
x = boston_df['DIS']

x = sm.add_constant(x)

results = sm.OLS(y,x).fit()

results.summary()

np.sqrt(0.062)

org_df.corr()

plt.figure(figsize=(16,9))
sns.heatmap(org_df.corr(),cmap="coolwarm",annot=True,fmt='.2f',linewidths=2, cbar=False)
plt.show()

