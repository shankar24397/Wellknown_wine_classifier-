
import imp
from sklearn import datasets
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split as tts
wine = datasets.load_wine()

features = wine.data
labels = wine.target


train_feats, test_feats, train_labels, test_labels = tts(features, labels, test_size=0.2)
'''
print("Number of Entries :", len(train_feats))


for featurename in wine.feature_names:
    print(featurename[:10], end="     \t")
print("Class")

for feature, label in zip(train_feats, train_labels):
    for f in feature:
        print(f, end="\t\t")
    print(label)
'''
#making classifier:
#this problem was solved using 3 different classifier techniques

#clf = svm.SVC(kernel="linear")
#above mthd used by default radial classification to seperate the class of classes.

clf= KNeighborsClassifier()

#clf = tree.DecisionTreeClassifier()
#clf = RandomForestClassifier()
#training the model



clf.fit(train_feats, train_labels)

#predictions
predictions = clf.predict(test_feats)
print(predictions)

score = 0
for i in range(len(predictions)):
    if(predictions[i] == test_labels[i]):
        score+=1
print("Accuracy:", (score/len(predictions))*100)









