# Databricks notebook source
import os
import requests
import numpy as np
import pandas as pd
import json

# COMMAND ----------

df_spark = spark.read.format("delta").load("/mnt/delta/PPG_ACC_dataset_pre")
df_pandas = df_spark.toPandas()

y_data = np.reshape(np.array(df_pandas["target"]), (-1,1))
df_pandas = df_pandas.drop(columns=["target"])

table = []
for index, row in df_pandas.iterrows():
    row = np.concatenate([row.tolist()],axis=0)
    table.append(row)
    
x_data = np.concatenate([table],axis=0)

# COMMAND ----------

test_data = x_data.tolist()[0:2]

# COMMAND ----------

url = 'https://mlab-masterthesis.cloud.databricks.com/model/rnn-model/2/invocations'
headers = {'Authorization': f'Bearer dapi8b42a6b3ce3cbc419a94f8203d6e48c3'}
response = requests.request(method='POST', headers=headers, url=url, json={'inputs': test_data})
if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')

# COMMAND ----------

response = response.json()

# COMMAND ----------

pred = np.argmax(response,axis=1)
pred
