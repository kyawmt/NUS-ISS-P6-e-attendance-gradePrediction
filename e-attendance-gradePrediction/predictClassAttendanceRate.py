#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd

import numpy as np

from prophet import Prophet

from prophet.diagnostics import cross_validation

from prophet.diagnostics import performance_metrics

from prophet.plot import plot_cross_validation_metric


# In[16]:


df = pd.read_csv("ProphetTest_01M019.csv")
df.head()


# In[17]:


m = Prophet()
m.fit(df)


# In[18]:


future = m.make_future_dataframe(periods=365)
future.tail()


# In[19]:


future


# In[20]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[21]:


fig1 = m.plot(forecast)


# In[22]:


fig2 = m.plot_components(forecast)


# In[28]:


df_cv = cross_validation(m, initial='67 days', period='90 days', horizon = '180 days')
df_cv.head()


# In[29]:


df_p = performance_metrics(df_cv)
df_p.head()


# In[30]:


fig = plot_cross_validation_metric(df_cv, metric='mape')
# The error of prediction is about 5%


# In[31]:


forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

