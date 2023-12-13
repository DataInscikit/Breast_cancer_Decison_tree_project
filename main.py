import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# loading the dataset
data = load_breast_cancer()
dataset = pd.DataFrame(data=data['data'], columns=data['feature_names'])

#view all the columns
pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 570)
pd.set_option('display.max_columns', 31)
print(dataset.head(5))

#partitioning the dataset into training and test datasets
from sklearn.model_selection import train_test_split
X = dataset.copy()
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#building the decision tree and training the model
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth = 4, ccp_alpha=0.01)
clf = clf.fit(X_train, y_train)

# predictions on the test dataset
print(clf.get_params())
predictions = clf.predict(X_test)
print(predictions)
print(clf.predict_proba(X_test))
print("\n")

#Evaluating the model's performance
from sklearn.metrics import accuracy_score
print("accuracy score: " + str(accuracy_score(y_test, predictions)))

from sklearn.metrics import precision_score
print("precision score: " + str(precision_score(y_test, predictions)))

from sklearn.metrics import recall_score
print("recall score: " + str(recall_score(y_test, predictions)))

from sklearn.metrics import confusion_matrix
print("confusion matrix: \n" + str(confusion_matrix(y_test, predictions,labels=[0,1])))

from sklearn.metrics import classification_report
print("classification report:\n " + str(classification_report(y_test, predictions, target_names = ['malignant','benign'])))

feature_importance = pd.DataFrame(clf.feature_importances_,index=X.columns).sort_values(0, ascending = False)
print(feature_importance)

#variable of importance chart
feature_importance.head(10).plot(kind='bar')
plt.title('Variable of importance chart')
plt.xlabel('Variables')
plt.ylabel('Value of importance')
plt.show()

#plotting decision tree
from sklearn import tree
fig = plt.figure(figsize = (100,10 ))
ref = tree.plot_tree(clf, feature_names=X.columns, class_names={0:'Malignant', 1:'Benign'}, filled=True, fontsize=12)
plt.title('Decision Tree for predicting breast cancer')
plt.show()
