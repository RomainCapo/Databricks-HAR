# Databricks notebook source
# MAGIC %md
# MAGIC # Data preprocessing
# MAGIC ## Setup input parameters

# COMMAND ----------

dbutils.widgets.text("window","1200","window")
dbutils.widgets.text("overlap","600","overlap")
dbutils.widgets.text("train_subjects","0-1-2-3","train_subjects")
dbutils.widgets.text("validation_subjects","4","validation_subjects")
dbutils.widgets.text("test_subjects","5-6","test_subjects")

# COMMAND ----------

# MAGIC %md
# MAGIC # Import libraries

# COMMAND ----------

import urllib 
import numpy as np
import scipy.io
import scipy.signal
import pandas as pd
import os
import shutil
import argparse

from sklearn.preprocessing import StandardScaler

# COMMAND ----------

# MAGIC %run "./HAR/HumanActivityRecognition"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define delta table path

# COMMAND ----------

DELTA_SILVER_PATH = "/tmp/delta/silver"
DELTA_GOLD_PATH = "/tmp/delta/gold"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze level raw data
# MAGIC Create table for raw data

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS bronze_raw_data (path STRING, modificationTime TIMESTAMP, length LONG)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data retrieval from the S3 mount point

# COMMAND ----------

# MAGIC %sql
# MAGIC COPY INTO bronze_raw_data
# MAGIC FROM (
# MAGIC   SELECT path, modificationTime, length
# MAGIC   FROM 'dbfs:/mnt/s3-mounted-input/PPG_ACC_dataset/'
# MAGIC )
# MAGIC FILEFORMAT = BINARYFILE
# MAGIC PATTERN = 'S[0-9]/*.mat'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver level cleaned data
# MAGIC ### Get some parameters

# COMMAND ----------

window = int(dbutils.widgets.get("window"))
overlap = int(dbutils.widgets.get("overlap"))
train_subject = tuple(map(int, dbutils.widgets.get("train_subjects")[0].split(" ")))
validation_subject = tuple(map(int, dbutils.widgets.get("validation_subjects")[0].split(" ")))
test_subject = tuple(map(int, dbutils.widgets.get("test_subjects")[0].split(" ")))

x_data, y_data, subj_inputs = create_dataset("/dbfs/mnt/s3-mounted-input/PPG_ACC_dataset", window, overlap)
clean_data(x_data)
x_data_norm = normalize(x_data)
x_data, y_data, subj_inputs = oversampling(x_data, y_data, subj_inputs, len(train_subject + validation_subject))

# COMMAND ----------

table = []
for x0 in range(x_data_norm.shape[0]):
    table.append([])
    for x1 in range(x_data_norm.shape[1]):
        table[x0].append(x_data_norm[x0,x1,:])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save PPG ACC dataset to delta table

# COMMAND ----------

table_name = "default.silver_ppg_acc_dataset"

path = DELTA_SILVER_PATH + "/PPG_ACC_dataset"

df_pandas_silver = pd.DataFrame(table)
df_pandas_silver["target"] = y_data
df_spark_silver = spark.createDataFrame(df_pandas_silver)
df_spark_silver.write.format("delta").mode("overwrite").save(path)

spark.sql("CREATE TABLE IF NOT EXISTS " + table_name + " USING DELTA LOCATION '" + path + "'")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save subjects index to delta table

# COMMAND ----------

table_name = "default.silver_ppg_acc_subjects_index"

path = DELTA_SILVER_PATH + "/subjects_index"

df_spark_subjects_index_silver = spark.createDataFrame(pd.DataFrame(subj_inputs, columns=["subjects_index"]))
df_spark_subjects_index_silver.write.format("delta").mode("overwrite").save(path)

spark.sql("CREATE TABLE IF NOT EXISTS " + table_name + " USING DELTA LOCATION '" + path + "'")
