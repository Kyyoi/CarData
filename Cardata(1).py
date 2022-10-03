#!/usr/bin/env python
# coding: utf-8

# In[227]:


import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
clf = tree.DecisionTreeClassifier()

df = pd.read_csv('/home/etudiant/Téléchargements/car.data', sep=",")
df = df.set_axis(['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety','class_value'],axis=1)
df.head()

X= df[['buying','maint','doors','persons','lug_boot', 'safety']]
y= df["class_value"]
X.head()

buy = {'vhigh': 4,'high': 3,'med':2,'small':1,'low':0}
safe={'high':3,'med':2,'low':1}
more= {'5more':6,'more':6,"2":2,"1":1,"3":3,"4":4}
target={"unacc":4,"acc":3,'good':2,"vgood":1}
lug={"low":1,'small':1,"med":3,'high':3,"big":5}

print(df)
""""
le = preprocessing.LabelEncoder() version avec preprocessing
le.fit(df.class_value)
list(le.classes_)
le.transform([df.class_value])"""


# In[228]:



df.buying = [buy[item] for item in df.buying]
df.maint = [buy[item] for item in df.maint]
df.doors = [more[item] for item in df.doors]
df.persons = [more[item] for item in df.persons]
df.lug_boot = [lug[item] for item in df.lug_boot]
df.safety = [safe[item] for item in df.safety]
df.class_value = [target[item] for item in df.class_value]
x= df[['buying','maint','doors','persons','lug_boot', 'safety']]
y= df["class_value"]
print(x)


# In[279]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.05,
random_state=0)


# In[280]:


clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)


# In[281]:


tree.plot_tree(clf, filled=True)


# In[282]:


clf.predict(X_test)
clf.score(X_test, y_test)


# In[283]:


tuned_parameters = {"min_samples_leaf":range(1,50),
                   "max_depth":range(1,50)}
grid= GridSearchCV(tree.DecisionTreeClassifier(),tuned_parameters,cv=2)

grid.fit(X_train, y_train)
print("Les meilleurs parametres:",grid.best_params_,
      "\nScore de test:",grid.score(X_test,y_test),
     "\nScore de train:",grid.score(X_train,y_train))


# In[277]:


import numpy as np
import matplotlib.pyplot as plt

# Paramètres
n_classes = 4
plot_colors = "bry" # blue-red-yellow
plot_step = 0.02

pair = [1, 2]

X=x.values[:,[0,2]]
print(x.columns)
y=y#class_values
print(x)

# Apprentissage de l'arbre
clf = tree.DecisionTreeClassifier().fit(X, y)
# Affichage de la surface de décision
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1


# In[268]:


y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min,
y_max, plot_step))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.xlabel("buying")
plt.ylabel("doors")
plt.axis("tight")
# Affichage des points d'apprentissage
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color,
    cmap=plt.cm.Paired)
plt.axis("tight")
plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend()
plt.show()


# In[ ]:




