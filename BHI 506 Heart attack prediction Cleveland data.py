#!/usr/bin/env python
# coding: utf-8

# # Predicting heart attack using machine learning

# ## Preprocessing Data

# ### There are some instances where the value is "?" which will be replaced with mean of that colum

# In[66]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(color_codes=True)


# In[67]:


# read the CSV file
data = pd.read_csv("processed.cleveland.csv", header=None)

# set column names
header = ['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall','output']
data.columns = header

# replace "?" with mean value of the corresponding column
for col in data.columns:
    if data[col].dtype != np.int64:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        mean_value = int(data[col].mean())
        data[col] = data[col].fillna(mean_value)
        data[col] = data[col].astype(int)
data.to_csv('processed.cleveland.csv', index=False)
# print the modified DataFrame
print(data)


# In[68]:


data.head()


# In[69]:


data.info()


# In[ ]:





# In[70]:


corr = data.corr()
corr


# In[71]:


plt.figure(figsize=(15, 10))
sns.heatmap(corr,
            annot=True)


# In[105]:


sns.pairplot(data[["age", "cp", "trtbps", 'chol', 'sex']])


# In[ ]:





# In[ ]:





# ### droping the unnecesary colums that has no relation

# In[73]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop("output", axis=1),
                                                    data["output"],
                                                    test_size=0.3)


# In[ ]:





# In[ ]:





# In[ ]:





# In[74]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # Random Forest Classifier

# In[75]:


from sklearn.ensemble import RandomForestClassifier


# In[76]:


forest = RandomForestClassifier()
forest.fit(X_train, y_train)
forest.score(X_test, y_test)


# In[77]:


forest_preds = forest.predict(X_test)
forest_preds


# In[92]:


from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, forest_preds))
sns.heatmap(confusion_matrix(y_test, forest_preds),
            annot=True)
plt.xlabel("Actual Labels")
plt.ylabel("Predicted Labels")


# In[93]:


print(classification_report(y_test, forest_preds))


# In[ ]:





# # Logistic Regression

# In[79]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=90)
log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)


# In[80]:


log_reg_preds = log_reg.predict(X_test)
log_reg_preds


# In[81]:


# Evaluate the prediced labels with the original labels
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, log_reg_preds))
sns.heatmap(confusion_matrix(y_test, log_reg_preds),
            annot=True)
plt.xlabel("Actual Labels")
plt.ylabel("Predicted Labels")


# In[82]:


print(classification_report(y_test, log_reg_preds))


# In[ ]:





# # Logistic Regression Model using the GridSearchCV

# In[94]:


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


# In[95]:


grid_model.fit(X_train, y_train)


# In[96]:


grid_model.best_params_


# In[97]:


grid_model.score(X_train, y_train)


# In[100]:


grid_preds = grid_model.predict(X_test)
grid_preds


# In[101]:


print(confusion_matrix(y_test, log_reg_preds))
sns.heatmap(confusion_matrix(y_test, grid_preds),
            annot=True)
plt.xlabel("Actual Labels")
plt.ylabel("Predicted Labels")


# In[102]:


print(classification_report(y_test, grid_preds))


# In[ ]:





# In[ ]:





# # Decision Tree Classifier

# In[106]:


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
dtree.score(X_test, y_test)


# In[107]:


dtree_preds = dtree.predict(X_test)
dtree_preds


# In[108]:


grid_preds = grid_model.predict(X_test)
grid_preds


# In[109]:


print(confusion_matrix(y_test, log_reg_preds))
sns.heatmap(confusion_matrix(y_test, grid_preds),
            annot=True)
plt.xlabel("Actual Labels")
plt.ylabel("Predicted Labels")


# In[110]:


print(classification_report(y_test, dtree_preds))


# In[ ]:




