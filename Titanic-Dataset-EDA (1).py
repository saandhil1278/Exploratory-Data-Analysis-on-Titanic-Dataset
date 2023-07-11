#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np;
import pandas as pd;


# In[31]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[32]:


df=pd.read_csv('train.csv')


# In[33]:


df


# df.shape

# In[34]:


df.columns.values


# 
# 

# In[35]:


df.info()


# In[36]:


df.isnull().sum()


# In[37]:


# Few conclusions

# 1. Missing values in Age, Cabin and Embarked columns
# 2. More than 70 percent values are missing in cabin columns, will have to drop
# 3. Few columns have inappropriate data types


# In[38]:


# Dropping cabin column

df.drop(columns=['Cabin'],inplace=True)


# In[39]:


# Imputing missing values for age
# Strategy - mean

df['Age'].fillna(df['Age'].mean(), inplace=True)


# In[40]:


# Imputing missing values for embarked

# finding the most appeared value in embarked column

df['Embarked'].value_counts()

# S it is

df['Embarked'].fillna('S', inplace=True)


# In[41]:


# Want to check one more thing...

# Should I change the SibSp and Parch to categories

df['SibSp'].value_counts()


# In[42]:


df['Parch'].value_counts()


# In[43]:


df['Survived']=df['Survived'].astype('category')
df['Pclass']=df['Pclass'].astype('category')
df['Sex']=df['Sex'].astype('category')
df['Age']=df['Age'].astype('int')
df['Embarked']=df['Embarked'].astype('category')


# In[44]:


df.info()


# In[45]:


# Five point summary
df.describe()


# In[46]:


# Univariate Analysis

# Let's start with the Survived col

sns.countplot(df['Survived'])

death_percent=round((df['Survived'].value_counts().values[0]/891)*100)

print("Out of 891 {} people died in the accident".format(death_percent))


# In[47]:


# Pclass column

print((df['Pclass'].value_counts()/891)*100)

sns.countplot(df['Pclass'])

# Conclusion : Pclass was the most crowded class


# In[48]:


print((df['Sex'].value_counts()/891)*100)

sns.countplot(df['Sex'])


# In[49]:


print(df['SibSp'].value_counts())

sns.countplot(df['SibSp'])


# In[50]:


print((df['Parch'].value_counts()/891)*100)

sns.countplot(df['Parch'])


# In[51]:


print((df['Embarked'].value_counts()/891)*100)

sns.countplot(df['Embarked'])


# In[52]:


# Age column

sns.distplot(df['Age'])

print(df['Age'].skew())

print(df['Age'].kurt())


# In[53]:


sns.boxplot(df['Age'])


# In[54]:


# Just out of curiosity

print("People with age in between 60 and 70 are",df[(df['Age']>60) & (df['Age']<70)].shape[0])
print("People with age greater than 70 and 75 are",df[(df['Age']>=70) & (df['Age']<=75)].shape[0])
print("People with age greater than 75 are",df[df['Age']>75].shape[0])

print('-'*50)

print("People with age between 0 and 1",df[df['Age']<1].shape[0])


# In[55]:


# Fare column

sns.distplot(df['Fare'])


# In[56]:


print(df['Fare'].skew())
print(df['Fare'].kurt())


# In[57]:


sns.boxplot(df['Fare'])


# In[58]:


print("People with fare in between $200 and $300",df[(df['Fare']>200) & (df['Fare']<300)].shape[0])
print("People with fare in greater than $300",df[df['Fare']>300].shape[0])


# In[59]:


# Multivariate Analysis

# Survival with Pclass

sns.countplot(df['Survived'], hue=df['Pclass'])

pd.crosstab(df['Pclass'], df['Survived']).apply(lambda r: round((r/r.sum())*100,1), axis=1)


# In[60]:


# Survival with Sex

sns.countplot(df['Survived'], hue=df['Sex'])

pd.crosstab(df['Sex'], df['Survived']).apply(lambda r: round((r/r.sum())*100,1), axis=1)


# In[61]:


# Survival with Embarked

sns.countplot(df['Survived'], hue=df['Embarked'])

pd.crosstab(df['Embarked'], df['Survived']).apply(lambda r: round((r/r.sum())*100,1), axis=1)


# In[62]:


# Survived with Age

plt.figure(figsize=(15,6))
sns.distplot(df[df['Survived']==0]['Age'])
sns.distplot(df[df['Survived']==1]['Age'])


# In[63]:


# Survived with Fare

plt.figure(figsize=(15,6))
sns.distplot(df[df['Survived']==0]['Fare'])
sns.distplot(df[df['Survived']==1]['Fare'])


# In[64]:


sns.pairplot(df)


# In[65]:


sns.heatmap(df.corr())


# In[66]:


# Feature Engineering

# We will create a new column by the name of family which will be the sum of SibSp and Parch cols

df['family_size']=df['Parch'] + df['SibSp']


# In[67]:


df.sample(5)


# In[68]:


# Now we will enginner a new feature by the name of family type

def family_type(number):
    if number==0:
        return "Alone"
    elif number>0 and number<=4:
        return "Medium"
    else:
        return "Large"


# In[69]:


df['family_type']=df['family_size'].apply(family_type)


# In[70]:


df.sample(5)


# In[71]:


# Dropping SibSp, Parch and family_size

df.drop(columns=['SibSp','Parch','family_size'],inplace=True)


# In[72]:


df.sample(5)


# In[73]:


pd.crosstab(df['family_type'], df['Survived']).apply(lambda r: round((r/r.sum())*100,1), axis=1)


# In[74]:


# handling outliers in age(Almost normal)

df=df[df['Age']<(df['Age'].mean() + 3 * df['Age'].std())]
df.shape


# In[75]:


# handling outliers from Fare column

# Finding quartiles

Q1= np.percentile(df['Fare'],25)
Q3= np.percentile(df['Fare'],75)

outlier_low=Q1 - 1.5 * (Q3 - Q1)
outlier_high=Q3 + 1.5 * (Q3 - Q1)

df=df[(df['Fare']>outlier_low) & (df['Fare']<outlier_high)]


# In[76]:


# One hot encoding

df.sample(4)

# Cols to be transformed are Pclass, Sex, Embarked, family_type

pd.get_dummies(data=df, columns=['Pclass','Sex','Embarked','family_type'], drop_first=True)


# In[77]:


df=pd.get_dummies(data=df, columns=['Pclass','Sex','Embarked','family_type'], drop_first=True)


# In[78]:


plt.figure(figsize=(15,6))
sns.heatmap(df.corr(), cmap='summer')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




