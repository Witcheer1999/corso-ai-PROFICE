import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

csv_file = "19_08\\esercizio1\\dataset\\AEP_hourly.csv"

df = pd.read_csv(csv_file, parse_dates=['Datetime'])

second_col = df.columns[1]
df['Consumption'] = df[second_col].apply(lambda x: "High" if x > df[second_col].median() else "Low")

df['hour'] = df['Datetime'].dt.hour
df['day'] = df['Datetime'].dt.day
df['month'] = df['Datetime'].dt.month

X = df[['hour', 'day', 'month']]
y = df['Consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

tree = DecisionTreeClassifier()

tree.fit(X_train, y_train)
tree.predict(X_test)

precision = precision_score(y_test, tree.predict(X_test), pos_label="High")
print(precision)

recall = recall_score(y_test, tree.predict(X_test), pos_label="High")
print(recall)

plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=X.columns, class_names=True)
plt.show()