#!/usr/bin/env python
# coding: utf-8

# # Predicting heart attack using machine learning

# ## Preprocessing Data

# ### There are some instances where the value is "?" which will be replaced with mean of that colum

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(color_codes=True)


# In[2]:


data = pd.read_csv("Data 506.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[38]:


corr = data.corr()
corr


# In[39]:


plt.figure(figsize=(15, 10))
sns.heatmap(corr,
            annot=True)


# In[40]:


sns.pairplot(data[["age", "cp", "trtbps", 'sex']])


# In[ ]:





# In[ ]:





# ### droping the unnecesary colums that has no relation

# In[33]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop("output", axis=1),
                                                    data["output"],
                                                    test_size=0.3)


# In[9]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # Random Forest Classifier

# In[10]:


from sklearn.ensemble import RandomForestClassifier


# In[11]:


forest = RandomForestClassifier()
forest.fit(X_train, y_train)
forest.score(X_test, y_test)


# In[12]:


forest_preds = forest.predict(X_test)
forest_preds


# In[13]:


from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, forest_preds))
sns.heatmap(confusion_matrix(y_test, forest_preds),
            annot=True)
plt.xlabel("Actual Labels")
plt.ylabel("Predicted Labels")


# In[14]:


print(classification_report(y_test, forest_preds))


# In[ ]:





# # Logistic Regression

# In[15]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=90)
log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)


# In[16]:


log_reg_preds = log_reg.predict(X_test)
log_reg_preds


# In[17]:


# Evaluate the prediced labels with the original labels
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, log_reg_preds))
sns.heatmap(confusion_matrix(y_test, log_reg_preds),
            annot=True)
plt.xlabel("Actual Labels")
plt.ylabel("Predicted Labels")


# In[18]:


print(classification_report(y_test, log_reg_preds))


# In[ ]:





# # Logistic Regression Model using the GridSearchCV

# In[19]:


from sklearn.model_selection import GridSearchCV

estimator = LogisticRegression()
params_grid = {"penalty": ['l1', 'l2', 'elasticnet'],
               "multi_class": ['auto', 'ovr', 'multinomial'],
               "max_iter": [100, 120, 80],
               "solver": ['newton-cg', 'lbfgs', 'liblinear']}

grid_model = GridSearchCV(estimator= estimator,
                          param_grid= params_grid,
                          verbose=True,
                          cv= 5,
                          n_jobs= -1)


# In[20]:


grid_model.fit(X_train, y_train)


# In[21]:


grid_model.best_params_


# In[22]:


grid_model.score(X_train, y_train)


# In[23]:


grid_preds = grid_model.predict(X_test)
grid_preds


# In[24]:


print(confusion_matrix(y_test, log_reg_preds))
sns.heatmap(confusion_matrix(y_test, grid_preds),
            annot=True)
plt.xlabel("Actual Labels")
plt.ylabel("Predicted Labels")


# In[25]:


print(classification_report(y_test, grid_preds))


# In[ ]:





# In[ ]:





# # Decision Tree Classifier

# In[26]:


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
dtree.score(X_test, y_test)


# In[27]:


dtree_preds = dtree.predict(X_test)
dtree_preds


# In[28]:


print(classification_report(y_test, dtree_preds))


# In[ ]:




