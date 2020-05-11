# Determining Wine Quality

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing Dataset
dataset = pd.read_csv('winequality-red.csv')
X = dataset.iloc[:, :10].values
y = dataset.iloc[:, 11].values


# Splitting into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# As the difference is high between variables, we apply feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Applying Model 
from sklearn.tree import DecisionTreeClassifier
regressor = DecisionTreeClassifier()
regressor.fit(X_train, y_train)

# Predicting results
y_pred = regressor.predict(X_test)

# Confidence Score
confidence = regressor.score(X_test, y_test)
print("\nThe confidence score:\n")
print(confidence)

# Visualising the Predicted Results
plt.hist(y_pred)
plt.show()

# Visualising the Original Test Results
plt.hist(y_test)
plt.show()



