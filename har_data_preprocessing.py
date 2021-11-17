# Databricks notebook source
dbutils.widgets.text("window","1200","window")
dbutils.widgets.text("overlap","600","overlap")

# COMMAND ----------

# MAGIC %md
# MAGIC # Import

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

DELTA_SILVER_PATH = "/tmp/delta/silver"
DELTA_GOLD_PATH = "/tmp/delta/gold"

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Bronze level raw data
# MAGIC Create table for raw data

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS bronze_raw_data (path STRING, modificationTime TIMESTAMP, length LONG)

# COMMAND ----------

# MAGIC %md
# MAGIC Data retrieval from the S3 mount point

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

def create_dataset(dataset_dir, window, overlap):
  """Create dataset as numpy array format from .mat file

    Args:
        dataset_dir (string): Directory where subjects folder are contained
        window (int): Sample length
        overlap (int): Window overlap

    Returns:
        tuple: Input data as numpy array format, Output data as numpy array format, Demarcation index of each subject in the numpy table 
    """

  # Create the input and target data from PPG_ACC_dataset,
  # according to window and overlap
  subj_list = [1, 2, 3, 4, 5, 6, 7]  # 1-based
  x_data = np.empty((0, window, 4))
  y_data = np.empty((0, 1))  # labels
  subj_inputs = []  # number of inputs for every subject
  subj_index = []

  # load data from PPG_ACC_dataset and reshape for RNN
  tot_rows = 0
  for subject in subj_list:
    subj_inputs.append(0)
    for category, name in enumerate(('rest', 'squat', 'step')):
      for record in range(0, 5):
        acc = scipy.io.loadmat(f'{dataset_dir}/S{subject}/{name}{record + 1}_acc.mat')['ACC']
        ppg = scipy.io.loadmat(f'{dataset_dir}/S{subject}/{name}{record + 1}_ppg.mat')['PPG'][:, 0:2]  # some PPG files have 3 columns instead of 2
        fusion = np.hstack((acc[:, 1:], ppg[:, 1:]))  # remove x axis (time)
        tot_rows += len(fusion)
        #clean_data(fusion)

        # windowing
        # compute number of windows (lazy way)
        i = 0
        num_w = 0
        while i + window  <= len(fusion):
          i += (window - overlap)
          num_w += 1
          subj_index.append(subject)
        # compute actual windows
        x_data_part = np.empty((num_w, window, 4))  # preallocate
        i = 0
        for w in range(0, num_w):
          x_data_part[w] = fusion[i:i + window]
          i += (window - overlap)
        x_data = np.vstack((x_data, x_data_part))
        y_data = np.vstack((y_data, np.full((num_w, 1), category)))
        subj_inputs[-1] += num_w

  return x_data, y_data, subj_inputs

# COMMAND ----------

def clean_data(x_data):
  """Clean input data. Replacement of the Nan values by an interpolated value of the two adjacent points.
    Replacement of zeros values by an interpolated value of the two adjacent points for the PPG.
    Replacement of some missing values by an interpolated value of the two adjacent points for the accelerometer 

    Args:
        x_data (np.array): Cleaned input data
  """

  for i in range(x_data.shape[0]):
    for col in range(0,4):
      ids = np.where(np.isnan(x_data[i,:, col]))[0]
      for row in ids:
        x_data[i, row, col] = 0.5 * (x_data[i, row - 1, col] + x_data[i, row + 1, col])

    for col in range(3, 4):
      ids = np.where(x_data[i,:, col] == 0)[0]
      for row in ids:
        x_data[i,row, col] = 0.5 * (x_data[i,row - 1, col] + x_data[i,row + 1, col])

    for col in range(0, 3):
      for row in range(1, x_data.shape[1] - 1):
        if abs(x_data[i,row, col] - x_data[i,row - 1, col]) > 5000 and abs(x_data[i,row, col] - x_data[i,row + 1, col]) > 5000:
          x_data[i,row, col] = 0.5 * (x_data[i,row - 1, col] + x_data[i,row + 1, col])

# COMMAND ----------

def normalize(x_data):
    """Normalize input data. Subtraction of the mean for the accelerometer components, z-norm for the PPG.  

    Args:
        x_data (np.array): Input data.

    Returns:
        np.array: Normalized data.
    """

    for w in x_data:
        # remove mean value from ACC
        w[:, 0] -= np.mean(w[:, 0])  # acc 1
        w[:, 1] -= np.mean(w[:, 1])  # acc 2
        w[:, 2] -= np.mean(w[:, 2])  # acc 3
        # standardize PPG
        w[:, 3] = StandardScaler().fit_transform(w[:, 3].reshape(-1, 1)).reshape((w.shape[0],))  # PPG

    return x_data

# COMMAND ----------

window = int(dbutils.widgets.get("window"))
overlap = int(dbutils.widgets.get("overlap"))

x_data, y_data, subj_inputs = create_dataset("/dbfs/mnt/s3-mounted-input/PPG_ACC_dataset", window, overlap)
clean_data(x_data)
x_data_norm = normalize(x_data)

# COMMAND ----------

table = []
for x0 in range(x_data_norm.shape[0]):
    table.append([])
    for x1 in range(x_data_norm.shape[1]):
        table[x0].append(x_data_norm[x0,x1,:])

# COMMAND ----------

table_name = "default.silver_ppg_acc_dataset"

path = DELTA_SILVER_PATH + "/PPG_ACC_dataset"

df_pandas_silver = pd.DataFrame(table)
df_pandas_silver["target"] = y_data
df_spark_silver = spark.createDataFrame(df_pandas_silver)
df_spark_silver.write.format("delta").mode("overwrite").save(path)

spark.sql("CREATE TABLE IF NOT EXISTS " + table_name + " USING DELTA LOCATION '" + path + "'")

# COMMAND ----------

table_name = "default.silver_ppg_acc_subjects_index"

path = DELTA_SILVER_PATH + "/subjects_index"

df_spark_subjects_index_silver = spark.createDataFrame(pd.DataFrame(subj_inputs, columns=["subjects_index"]))
df_spark_subjects_index_silver.write.format("delta").mode("overwrite").save(path)

spark.sql("CREATE TABLE IF NOT EXISTS " + table_name + " USING DELTA LOCATION '" + path + "'")
