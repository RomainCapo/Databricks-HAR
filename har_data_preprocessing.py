# Databricks notebook source
# MAGIC %md
# MAGIC # Data preprocessing
# MAGIC ## Setup input parameters

# COMMAND ----------

dbutils.widgets.text("window","1200","window")
dbutils.widgets.text("overlap","600","overlap")
dbutils.widgets.text("train_subjects","0-1-2-3","train_subjects")
dbutils.widgets.text("validation_subjects","4","validation_subjects")
dbutils.widgets.text("classes","rest-squat-step","classes")
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

DELTA_LAKE_PATH= "/tmp/delta/dataset"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver level cleaned data
# MAGIC ### Get some parameters

# COMMAND ----------

window = int(dbutils.widgets.get("window"))
overlap = int(dbutils.widgets.get("overlap"))
train_subject = tuple(map(int, dbutils.widgets.get("train_subjects").split("-")))
validation_subject = tuple(map(int, dbutils.widgets.get("validation_subjects").split("-")))
test_subject = tuple(map(int, dbutils.widgets.get("test_subjects").split("-")))
classes = dbutils.widgets.get("classes").split("-")

dataset_dir = "/dbfs/mnt/s3-mounted-input/PPG_ACC_dataset"

subject_index = list(range(0, len(os.listdir(dataset_dir))))

x_data, y_data, subj_inputs = create_dataset(dataset_dir, subject_index, classes, window, overlap)
clean_data(x_data)
x_data_norm = normalize(x_data)
x_data_norm, y_data, subj_inputs = oversampling(x_data_norm, y_data, subj_inputs, len(train_subject + validation_subject))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert 3D dataset to 2D by storing vector

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

table_name = "default.ppg_acc_dataset"

path = DELTA_LAKE_PATH + "/PPG_ACC_dataset"

df_pandas_ppg_acc = pd.DataFrame(table)
df_pandas_ppg_acc["target"] = y_data
df_spark_ppg_acc = spark.createDataFrame(df_pandas_ppg_acc)
df_spark_ppg_acc.write.format("delta").mode("overwrite").save(path)

spark.sql("CREATE TABLE IF NOT EXISTS " + table_name + " USING DELTA LOCATION '" + path + "'")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save subjects index to delta table

# COMMAND ----------

table_name = "default.ppg_acc_subjects_index"

path = DELTA_LAKE_PATH + "/subjects_index"

df_spark_subjects_index = spark.createDataFrame(pd.DataFrame(subj_inputs, columns=["subjects_index"]))
df_spark_subjects_index.write.format("delta").mode("overwrite").save(path)

spark.sql("CREATE TABLE IF NOT EXISTS " + table_name + " USING DELTA LOCATION '" + path + "'")

# COMMAND ----------


