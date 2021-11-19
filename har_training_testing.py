# Databricks notebook source
# MAGIC %md
# MAGIC # Model training and testing
# MAGIC ## Setup input parameters

# COMMAND ----------

dbutils.widgets.text("epochs","1","epochs")
dbutils.widgets.text("learning_rate","0.01","learning_rate")
dbutils.widgets.text("train_subjects","0-1-2-3","train_subjects")
dbutils.widgets.text("validation_subjects","4","validation_subjects")
dbutils.widgets.text("test_subjects","5-6","test_subjects")
dbutils.widgets.text("num_cell_dense1","32","num_cell_dense1")
dbutils.widgets.text("num_cell_lstm1","32","num_cell_lstm1")
dbutils.widgets.text("num_cell_lstm2","32","num_cell_lstm2")
dbutils.widgets.text("num_cell_lstm3","32","num_cell_lstm3")
dbutils.widgets.text("dropout_rate","0.01","dropout_rate")
dbutils.widgets.text("window","1200","window")
dbutils.widgets.text("overlap","600","overlap")
dbutils.widgets.dropdown("environment", "Development", ["Development","Staging","Production"])
dbutils.widgets.text("accuracy_thresold","0.6","accuracy_thresold")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install libraries

# COMMAND ----------

dbutils.library.installPyPI("mlflow")
dbutils.library.installPyPI("tensorflow")
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import libraries

# COMMAND ----------

import numpy as np
import mlflow
import tensorflow as tf
import unittest

from mlflow.tracking import client
from mlflow.models.signature import infer_signature
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LSTM, Dropout, InputLayer
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# COMMAND ----------

# MAGIC %run "./HAR/HumanActivityRecognition"

# COMMAND ----------

# MAGIC %run "./test/test"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data from delta table

# COMMAND ----------

df_spark = spark.read.format("delta").load("/tmp/delta/silver/PPG_ACC_dataset")
df_pandas = df_spark.toPandas()

# COMMAND ----------

df_spark_subjects = spark.read.format("delta").load("/tmp/delta/silver/subjects_index")
df_pandas_subjects = df_spark_subjects.toPandas()
subj_inputs = df_pandas_subjects['subjects_index'].tolist()

# COMMAND ----------

y_data = np.reshape(np.array(df_pandas["target"]), (-1,1))
df_pandas = df_pandas.drop(columns=["target"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reconstitution of the dataset

# COMMAND ----------

table = []
for index, row in df_pandas.iterrows():
    row = np.concatenate([row.tolist()],axis=0)
    table.append(row)
    
x_data = np.concatenate([table],axis=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Definition of model

# COMMAND ----------

def create_model(hyperparameters, num_classes, num_features):
    """Create LSTM model.

    Args:
        args (argparse): Argparse arguments objects.
        num_classes (int): Number of classes.
        num_features (int): Number of input features

    Returns:
        tuple: Partionned input data, Partionned output data
    """
        
    model = Sequential()
    
    model.add(InputLayer(input_shape=(hyperparameters["window"], num_features)))
    model.add(Dense(hyperparameters["num_cell_dense1"], name='dense1'))
    model.add(BatchNormalization(name='norm'))
    
    model.add(LSTM(hyperparameters["num_cell_lstm1"], return_sequences=True, name='lstm1'))
    model.add(Dropout(hyperparameters["dropout_rate"], name='drop2'))
    
    model.add(LSTM(hyperparameters["num_cell_lstm2"], return_sequences=True, name='lstm2'))
    model.add(Dropout(hyperparameters["dropout_rate"], name='drop3'))
    
    model.add(LSTM(hyperparameters["num_cell_lstm3"], name='lstm3'))
    model.add(Dropout(0.5, name='drop4'))
    
    model.add(Dense(num_classes, name='dense2')) 
    
    optimizer = Adam(hyperparameters["learning_rate"])
    
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer=optimizer, metrics=['accuracy'])
    return model


# COMMAND ----------

# MAGIC %md
# MAGIC ## Get some parameters and hyperparameters

# COMMAND ----------

num_classes = len(set(y_data.flatten()))
num_features = x_data.shape[2]

accuracy_thresold = float(dbutils.widgets.get("accuracy_thresold"))
window = int(dbutils.widgets.get("window"))
environment = dbutils.widgets.get("environment")

hyperparameters = {
    "epochs":int(dbutils.widgets.get("epochs")),
    "learning_rate":float(dbutils.widgets.get("learning_rate")),
    "train_subjects":tuple(map(int, dbutils.widgets.get("train_subjects")[0].split(" "))),
    "validation_subjects": tuple(map(int, dbutils.widgets.get("validation_subjects")[0].split(" "))),
    "test_subjects": tuple(map(int, dbutils.widgets.get("test_subjects")[0].split(" "))),
    "num_cell_dense1":int(dbutils.widgets.get("num_cell_dense1")),
    "num_cell_lstm1":int(dbutils.widgets.get("num_cell_lstm1")),
    "num_cell_lstm2":int(dbutils.widgets.get("num_cell_lstm2")),
    "num_cell_lstm3":int(dbutils.widgets.get("num_cell_lstm3")),
    "dropout_rate":float(dbutils.widgets.get("dropout_rate")),
    "window":window,
    "overlap":int(dbutils.widgets.get("overlap"))
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split dataset

# COMMAND ----------

x_data_train, y_data_train = partition_data(hyperparameters["train_subjects"], subj_inputs, x_data, y_data)
x_data_validation, y_data_validation = partition_data(hyperparameters["validation_subjects"], subj_inputs, x_data, y_data)
x_data_test, y_data_test = partition_data(hyperparameters["test_subjects"], subj_inputs, x_data, y_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define function to promotes new model

# COMMAND ----------

def promotes_new_model(stage, model_name):
    """Archive all model wih the given stage and promotes the last one.

    Args:
        stage (string): Model stage
        model_name (string): Model name
    """
    mlflowclient = client.MlflowClient()
    max_version = 0

    for mv in mlflowclient.search_model_versions(f"name='{model_name}'"):
        current_version = int(dict(mv)['version'])
        if current_version > max_version:
            max_version = current_version
        if dict(mv)['current_stage'] == stage:
            version = dict(mv)['version']                                   
            mlflowclient.transition_model_version_stage(model_name, version, stage="Archived")

    mlflowclient.transition_model_version_stage(model_name, max_version, stage=stage)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train model and log with MLFlow

# COMMAND ----------

model_name = "rnn-model"
mlflow.set_experiment("/Repos/Production/Databricks-HAR/har_training")

with mlflow.start_run(run_name='lstm_har') as run:
    mlflow.log_param("epochs", hyperparameters["epochs"])
    mlflow.log_param("learning_rate", hyperparameters["learning_rate"])
    mlflow.log_param("train_subjects", hyperparameters["train_subjects"])
    mlflow.log_param("validation_subjects", hyperparameters["validation_subjects"])
    mlflow.log_param("test_subjects", hyperparameters["test_subjects"])
    mlflow.log_param("num_cell_dense1", hyperparameters["num_cell_dense1"])
    mlflow.log_param("num_cell_lstm1", hyperparameters["num_cell_lstm1"])
    mlflow.log_param("num_cell_lstm2", hyperparameters["num_cell_lstm2"])
    mlflow.log_param("num_cell_lstm3", hyperparameters["num_cell_lstm3"])
    mlflow.log_param("dropout_rate", hyperparameters["dropout_rate"])
    mlflow.log_param("window", hyperparameters["window"])
    mlflow.log_param("overlap", hyperparameters["overlap"])
    mlflow.log_param("environment", environment)
    mlflow.log_param("accuracy_thresold", accuracy_thresold)

    model = create_model(hyperparameters, num_classes, num_features)
    
    history = model.fit(x_data_train, y_data_train, epochs=hyperparameters["epochs"], validation_data=(x_data_validation, y_data_validation), steps_per_epoch=5)
    
    y_pred_proba = model.predict(x_data_test)
    y_pred = np.argmax(y_pred_proba,axis=1)

    accuracy = accuracy_score(y_data_test, y_pred)
    precision = precision_score(y_data_test, y_pred, average='weighted')
    recall = recall_score(y_data_test, y_pred, average='weighted')
    f1 = f1_score(y_data_test, y_pred, average='weighted')
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    
    # Execute test (see module test/test.py)
    test_names = ['test_shape_input', 'test_shape_output', 'test_model_input', 'test_model_output', 'test_prediction_shape']
    test_result = execute_test(test_names)
    
    if not False in test_result and accuracy >= accuracy_thresold: 
        signature = infer_signature(x_data_test, model.predict(x_data_test))

        input_example = np.expand_dims(x_data_train[0], axis=0)

        mlflow.keras.log_model(model, model_name, signature=signature, input_example=input_example, registered_model_name=model_name)
        
        promotes_new_model(environment, model_name)
    else :
        print("The model is not registered because the unit tests failed or the model did not reach the requested accuracy.")
