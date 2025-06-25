import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/zeebu/OneDrive/Documents/Python/Decision-Tree/train.csv")

d = {'male':0,'female':1}
df['Sex'] = df['Sex'].map(d)
d= {'C':0, 'Q':1, 'S':2}
df['Embarked'] = df['Embarked'].map(d)

features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

X = df[features]
y = df['Survived']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X,y)

tree.plot_tree(dtree, feature_names=features)



test = pd.read_csv("C:/Users/zeebu/OneDrive/Documents/Python/Decision-Tree/test.csv")

d = {'male':0,'female':1}
test['Sex'] = test['Sex'].map(d)
d= {'C':0, 'Q':1, 'S':2}
test['Embarked'] = test['Embarked'].map(d)

features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

test_features = test[features]

predictions = dtree.predict(test_features)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


