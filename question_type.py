import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer as cv
accuracy=[]
headers=["Questions","Types"]
df=pd.read_table("LabelledData.txt",delimiter=',,,',engine='python',names=headers)   # engine is python

print(df["Types"].unique())
print(df["Types"].value_counts())

corr=df.corr()
#check if any nan data in columns
for i in df.columns:
    print("nan data in" +" "+i+"=",df[i].isnull().sum())
    
print("\n")
#after remove the space


df["Types"]=df["Types"].str.lstrip()#in data set had extra space then remove.
#frequency count for unique type
count=df["Types"].value_counts()
print(count)
word_count=[]
word_count+=[i.split() for i in df.Questions.values]

print(df["Questions"].keys())
op=df["Questions"].values



vect=cv(analyzer="word",lowercase=True,tokenizer=None)
features=vect.fit_transform(op)#feature data /independent

labels=df["Types"]#target data/dependent


import numpy as np


print(features.shape[0])

array=np.arange(features.shape[0])
np.random.shuffle(array)
features=features[array].toarray()
labels=np.array(labels[array])


size=0.2*features.shape[0]

shape1=features.shape[0]-size

feature_train,feature_test,label_train,label_test=features[:int(shape1)],features[int(shape1):],labels[:int(shape1)],labels[int(shape1):]


#check best accuracy classifier
#naive bayes
from sklearn.naive_bayes import GaussianNB,MultinomialNB
clf=GaussianNB()
clf.fit(feature_train,label_train)
pre=clf.predict(feature_test)
from sklearn.metrics import accuracy_score
score=accuracy_score(label_test,pre)
print("%.2f Gaussian accuracy "%(100*score))

accuracy.append(score*100)
nb=MultinomialNB()
nb.fit(feature_train,label_train)
pre1=nb.predict(feature_test)
score1=accuracy_score(label_test,pre1)
print("%.2f Mutlinomial accuracy "%(100*score1))
accuracy.append(score1*100)

#decision tree classifier

from sklearn.tree import DecisionTreeClassifier
dc=DecisionTreeClassifier()
dc.fit(feature_train,label_train)
pre_dc=dc.predict(feature_test)
acc_dc=accuracy_score(label_test,pre_dc)
print("%.2f Decision tree accuracy"%(100*acc_dc))
accuracy.append(acc_dc*100)




model=["GaussianNB","MultinomialNB","DecisionTree"]
data_frame=pd.DataFrame({"model-name":model,"accuracy":accuracy})
acc_index=accuracy.index(max(accuracy))

model_selection=model[acc_index]
accuracy_selection=accuracy[acc_index]

print("\n"+"The final model selection {} based on accuracy is {} ".format(model_selection,accuracy_selection))

#test unseen data
while True:
    check=input("enter statement for checking feature unseen data")
    check=vect.transform([check])

    res_data=dc.predict(check)
    print("type:",res_data)

