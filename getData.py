# Import kaggle API module and get titanic competition data
from pathlib import Path

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# %%
train_df = pd.read_csv(Path().joinpath('data', 'train.csv'))
test_df = pd.read_csv(Path().joinpath('data', 'test.csv'))
combined_df = [train_df, test_df]

# Exploratory analysis
train_df.head()
train_df.tail()

# %%
sample_titanic = train_df.sample(10)
# %% Plotting

sns.countplot(x='Survived', hue='Sex', data=train_df)
plt.show()

# %%
sns.countplot(x='Survived', hue='Pclass', data=train_df)
plt.show()
