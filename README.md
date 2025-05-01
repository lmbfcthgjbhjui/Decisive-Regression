# Decisive-Regression
Here is the python code below :
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import graphviz
import pydotplus
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
Load the data :
# Load the dataset
data = pd.read_csv('heart.csv')

# Separate features (X) and target (y)
X = data.drop('target', axis=1)
y = data['target']
Data loaded and features (X) and the target variable ('target', which I assume indicates the presence or absence of heart disease) separated.
Now, let's follow the hints. First up: train a Decision Tree Classifier and visualize it. We'll split the data into training and testing sets first.
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree Classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_tree = tree_clf.predict(X_test)

# Evaluate the Decision Tree
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_tree))
Alright, let's dive into the world of decision trees and random forests! This is a fun part of machine learning where we get to see how these models make decisions, almost like a flowchart. Clicks the "click here to download dataset" link. Looks like we've got the Heart Disease dataset. Perfect for a classification task!

Now, let's get our hands coding. We'll need the usual suspects: pandas for data wrangling, scikit-learn for the models and evaluation, and Graphviz (plus pydotplus) for visualizing the decision tree.

Python

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import graphviz
import pydotplus
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
Okay, libraries imported. Let's load the data:

Python

# Load the dataset
data = pd.read_csv('heart.csv')

# Separate features (X) and target (y)
X = data.drop('target', axis=1)
y = data['target']
Data loaded and features (X) and the target variable ('target', which I assume indicates the presence or absence of heart disease) separated.

Now, let's follow the hints. First up: train a Decision Tree Classifier and visualize it. We'll split the data into training and testing sets first.

Python

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree Classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_tree = tree_clf.predict(X_test)

# Evaluate the Decision Tree
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
We've trained a basic decision tree and evaluated its performance. Now for the visualization. This requires exporting the tree structure into a format that Graphviz can understand.
# Visualize the Decision Tree
dot_data = StringIO()
export_graphviz(
    tree_clf,
    out_file=dot_data,
    feature_names=X.columns,
    class_names=['No Heart Disease', 'Heart Disease'],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('decision_tree.png')

# To display in a Jupyter Notebook (optional)
# from IPython.display import Image
# Image(graph.create_png())
Alright, let's dive into the world of decision trees and random forests! This is a fun part of machine learning where we get to see how these models make decisions, almost like a flowchart. Clicks the "click here to download dataset" link. Looks like we've got the Heart Disease dataset. Perfect for a classification task!

Now, let's get our hands coding. We'll need the usual suspects: pandas for data wrangling, scikit-learn for the models and evaluation, and Graphviz (plus pydotplus) for visualizing the decision tree.

Python

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import graphviz
import pydotplus
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
Okay, libraries imported. Let's load the data:

Python

# Load the dataset
data = pd.read_csv('heart.csv')

# Separate features (X) and target (y)
X = data.drop('target', axis=1)
y = data['target']
Data loaded and features (X) and the target variable ('target', which I assume indicates the presence or absence of heart disease) separated.

Now, let's follow the hints. First up: train a Decision Tree Classifier and visualize it. We'll split the data into training and testing sets first.

Python

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree Classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_tree = tree_clf.predict(X_test)

# Evaluate the Decision Tree
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_tree))
We've trained a basic decision tree and evaluated its performance. Now for the visualization. This requires exporting the tree structure into a format that Graphviz can understand.

Python

# Visualize the Decision Tree
dot_data = StringIO()
export_graphviz(
    tree_clf,
    out_file=dot_data,
    feature_names=X.columns,
    class_names=['No Heart Disease', 'Heart Disease'],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('decision_tree.png')

# To display in a Jupyter Notebook (optional)
# from IPython.display import Image
# Image(graph.create_png())
This code exports the decision tree structure to a .dot format, then uses pydotplus and Graphviz to render it as a PNG image named decision_tree.png. If you're in a Jupyter Notebook, you can uncomment the last two lines to display it directly. Looking at this image will show us the sequence of decisions the tree makes to classify a patient.

Next, let's analyze overfitting and control the tree depth. Decision trees are prone to overfitting, meaning they learn the training data too well, including the noise, and perform poorly on unseen data. One way to control this is by limiting the maximum depth of the tree.
# Train a Decision Tree with a limited depth
tree_clf_depth = DecisionTreeClassifier(max_depth=3, random_state=42) # Limiting to a depth of 3
tree_clf_depth.fit(X_train, y_train)
y_pred_tree_depth = tree_clf_depth.predict(X_test)
print("\nDecision Tree Accuracy (Max Depth 3):", accuracy_score(y_test, y_pred_tree_depth))
print("Decision Tree Confusion Matrix (Max Depth 3):\n", confusion_matrix(y_test, y_pred_tree_depth))
print("Decision Tree Classification Report (Max Depth 3):\n", classification_report(y_test, y_pred_tree_depth))

# Visualize the limited depth Decision Tree
dot_data_depth = StringIO()
export_graphviz(
    tree_clf_depth,
    out_file=dot_data_depth,
    feature_names=X.columns,
    class_names=['No Heart Disease', 'Heart Disease'],
    filled=True,
    rounded=True,
    special_characters=True
)
graph_depth = pydotplus.graph_from_dot_data(dot_data_depth.getvalue())
graph_depth.write_png('decision_tree_depth_3.png')
Alright, let's dive into the world of decision trees and random forests! This is a fun part of machine learning where we get to see how these models make decisions, almost like a flowchart. Clicks the "click here to download dataset" link. Looks like we've got the Heart Disease dataset. Perfect for a classification task!

Now, let's get our hands coding. We'll need the usual suspects: pandas for data wrangling, scikit-learn for the models and evaluation, and Graphviz (plus pydotplus) for visualizing the decision tree.

Python

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import graphviz
import pydotplus
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
Okay, libraries imported. Let's load the data:

Python

# Load the dataset
data = pd.read_csv('heart.csv')

# Separate features (X) and target (y)
X = data.drop('target', axis=1)
y = data['target']
Data loaded and features (X) and the target variable ('target', which I assume indicates the presence or absence of heart disease) separated.

Now, let's follow the hints. First up: train a Decision Tree Classifier and visualize it. We'll split the data into training and testing sets first.

Python

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree Classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_tree = tree_clf.predict(X_test)

# Evaluate the Decision Tree
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_tree))
We've trained a basic decision tree and evaluated its performance. Now for the visualization. This requires exporting the tree structure into a format that Graphviz can understand.

Python

# Visualize the Decision Tree
dot_data = StringIO()
export_graphviz(
    tree_clf,
    out_file=dot_data,
    feature_names=X.columns,
    class_names=['No Heart Disease', 'Heart Disease'],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('decision_tree.png')

# To display in a Jupyter Notebook (optional)
# from IPython.display import Image
# Image(graph.create_png())
This code exports the decision tree structure to a .dot format, then uses pydotplus and Graphviz to render it as a PNG image named decision_tree.png. If you're in a Jupyter Notebook, you can uncomment the last two lines to display it directly. Looking at this image will show us the sequence of decisions the tree makes to classify a patient.

Next, let's analyze overfitting and control the tree depth. Decision trees are prone to overfitting, meaning they learn the training data too well, including the noise, and perform poorly on unseen data. One way to control this is by limiting the maximum depth of the tree.

Python

# Train a Decision Tree with a limited depth
tree_clf_depth = DecisionTreeClassifier(max_depth=3, random_state=42) # Limiting to a depth of 3
tree_clf_depth.fit(X_train, y_train)
y_pred_tree_depth = tree_clf_depth.predict(X_test)
print("\nDecision Tree Accuracy (Max Depth 3):", accuracy_score(y_test, y_pred_tree_depth))
print("Decision Tree Confusion Matrix (Max Depth 3):\n", confusion_matrix(y_test, y_pred_tree_depth))
print("Decision Tree Classification Report (Max Depth 3):\n", classification_report(y_test, y_pred_tree_depth))

# Visualize the limited depth Decision Tree
dot_data_depth = StringIO()
export_graphviz(
    tree_clf_depth,
    out_file=dot_data_depth,
    feature_names=X.columns,
    class_names=['No Heart Disease', 'Heart Disease'],
    filled=True,
    rounded=True,
    special_characters=True
)
graph_depth = pydotplus.graph_from_dot_data(dot_data_depth.getvalue())
graph_depth.write_png('decision_tree_depth_3.png')
By setting max_depth=3, we've created a shallower tree. Comparing the performance of this tree with the original one will give us an idea of how limiting depth can affect generalization. Visualizing this shallower tree (decision_tree_depth_3.png) will also be much easier to interpret.

Now, let's move on to training a Random Forest and comparing its accuracy. Random Forests are an ensemble method that builds multiple decision trees and merges their predictions. They often perform better than single decision trees and are less prone to overfitting.
# Train a Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42) # Using 100 trees
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
Alright, let's dive into the world of decision trees and random forests! This is a fun part of machine learning where we get to see how these models make decisions, almost like a flowchart. Clicks the "click here to download dataset" link. Looks like we've got the Heart Disease dataset. Perfect for a classification task!

Now, let's get our hands coding. We'll need the usual suspects: pandas for data wrangling, scikit-learn for the models and evaluation, and Graphviz (plus pydotplus) for visualizing the decision tree.

Python

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import graphviz
import pydotplus
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
Okay, libraries imported. Let's load the data:

Python

# Load the dataset
data = pd.read_csv('heart.csv')

# Separate features (X) and target (y)
X = data.drop('target', axis=1)
y = data['target']
Data loaded and features (X) and the target variable ('target', which I assume indicates the presence or absence of heart disease) separated.

Now, let's follow the hints. First up: train a Decision Tree Classifier and visualize it. We'll split the data into training and testing sets first.

Python

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree Classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_tree = tree_clf.predict(X_test)

# Evaluate the Decision Tree
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_tree))
We've trained a basic decision tree and evaluated its performance. Now for the visualization. This requires exporting the tree structure into a format that Graphviz can understand.

Python

# Visualize the Decision Tree
dot_data = StringIO()
export_graphviz(
    tree_clf,
    out_file=dot_data,
    feature_names=X.columns,
    class_names=['No Heart Disease', 'Heart Disease'],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('decision_tree.png')

# To display in a Jupyter Notebook (optional)
# from IPython.display import Image
# Image(graph.create_png())
This code exports the decision tree structure to a .dot format, then uses pydotplus and Graphviz to render it as a PNG image named decision_tree.png. If you're in a Jupyter Notebook, you can uncomment the last two lines to display it directly. Looking at this image will show us the sequence of decisions the tree makes to classify a patient.

Next, let's analyze overfitting and control the tree depth. Decision trees are prone to overfitting, meaning they learn the training data too well, including the noise, and perform poorly on unseen data. One way to control this is by limiting the maximum depth of the tree.

Python

# Train a Decision Tree with a limited depth
tree_clf_depth = DecisionTreeClassifier(max_depth=3, random_state=42) # Limiting to a depth of 3
tree_clf_depth.fit(X_train, y_train)
y_pred_tree_depth = tree_clf_depth.predict(X_test)
print("\nDecision Tree Accuracy (Max Depth 3):", accuracy_score(y_test, y_pred_tree_depth))
print("Decision Tree Confusion Matrix (Max Depth 3):\n", confusion_matrix(y_test, y_pred_tree_depth))
print("Decision Tree Classification Report (Max Depth 3):\n", classification_report(y_test, y_pred_tree_depth))

# Visualize the limited depth Decision Tree
dot_data_depth = StringIO()
export_graphviz(
    tree_clf_depth,
    out_file=dot_data_depth,
    feature_names=X.columns,
    class_names=['No Heart Disease', 'Heart Disease'],
    filled=True,
    rounded=True,
    special_characters=True
)
graph_depth = pydotplus.graph_from_dot_data(dot_data_depth.getvalue())
graph_depth.write_png('decision_tree_depth_3.png')
By setting max_depth=3, we've created a shallower tree. Comparing the performance of this tree with the original one will give us an idea of how limiting depth can affect generalization. Visualizing this shallower tree (decision_tree_depth_3.png) will also be much easier to interpret.

Now, let's move on to training a Random Forest and comparing its accuracy. Random Forests are an ensemble method that builds multiple decision trees and merges their predictions. They often perform better than single decision trees and are less prone to overfitting.

Python

# Train a Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42) # Using 100 trees
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
We've trained a Random Forest with 100 trees (n_estimators=100) and evaluated its accuracy. Let's see if it's better than our single decision tree.

Next up: interpret feature importances. Random Forests provide a way to assess which features were most important in making predictions.
# Get feature importances from the Random Forest
importances = rf_clf.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances in Random Forest")
plt.bar(range(X.shape[1]), importances[sorted_indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()
Finally Cross Validation :
# Cross-validation for Decision Tree
cv_scores_tree = cross_val_score(tree_clf, X, y, cv=5, scoring='accuracy')
print("\nDecision Tree Cross-Validation Scores:", cv_scores_tree)
print("Decision Tree Average Cross-Validation Accuracy:", np.mean(cv_scores_tree))

# Cross-validation for Random Forest
cv_scores_rf = cross_val_score(rf_clf, X, y, cv=5, scoring='accuracy')
print("Random Forest Cross-Validation Scores:", cv_scores_rf)
print("Random Forest Average Cross-Validation Accuracy:", np.mean(cv_scores_rf))
