from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz 
import pandas as pd

with open('Data/adult.data', newline='') as csvfile:
    data = pd.read_csv(csvfile, sep=',')

x=data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
y=data.iloc[:,[14]]


le = preprocessing.LabelEncoder()
y=le.fit_transform(y.values.ravel())
oe = preprocessing.OrdinalEncoder()
x=oe.fit_transform(x)

xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size=0.2, random_state=50)

clf = tree.DecisionTreeClassifier(min_samples_leaf=30, max_depth=8)
clf = clf.fit(xTrain, yTrain)

yPred=clf.predict(xTest) 
accuracy=accuracy_score(yTest, yPred)
print(str(accuracy*100)+"% accurate")

categories=["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country"]

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=categories, class_names=['<=50k', '>50k'], filled=True)
graph = graphviz.Source(dot_data)  
graph.render("q2")