# Databricks notebook source
dbutils.widgets.text("epochs","1","epochs")
dbutils.widgets.text("learning_rate","0.01","learning_rate")
dbutils.widgets.text("train_subjects","0-1-2-3","train_subjects")
dbutils.widgets.text("validation_subjects","4","validation_subjects")
dbutils.widgets.text("test_subjects","5-6","test_subjects")
dbutils.widgets.text("num_cell_dense1","32","num_cell_dense1")
dbutils.widgets.text("num_cell_lstm1","1","num_cell_lstm1")
dbutils.widgets.text("window","1200","window")
dbutils.widgets.text("overlap","600","overlap")


# COMMAND ----------


1
hyperparameters = {
2
    "epochs":1,
3
    "learning_rate":0.01,
4
    "train_subjects":(0,1,2,3),
5
    "validation_subjects": (4,),
6
    "test_subjects": (5,6),
7
    "num_cell_dense1":32,
8
    "num_cell_lstm1":32,
9
    "num_cell_lstm2":32,
10
    "num_cell_lstm3":32,
11
    "dropout_rate":0.2,
12
    "window":1200,
13
    "overlap":600
14
}

# COMMAND ----------

dbutils.library.installPyPI("mlflow")
dbutils.library.installPyPI("tensorflow")
dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import mlflow
import tensorflow as tf

from mlflow.models.signature import infer_signature
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, BatchNormalization, LSTM, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# COMMAND ----------

df_spark = spark.read.format("delta").load("/mnt/delta/PPG_ACC_dataset_pre")
df_pandas = df_spark.toPandas()

# COMMAND ----------

subj_inputs = [2663, 2364, 1193, 1205, 1264, 1288, 1333]

# COMMAND ----------

y_data = np.reshape(np.array(df_pandas["target"]), (-1,1))
df_pandas = df_pandas.drop(columns=["target"])

# COMMAND ----------

table = []
for index, row in df_pandas.iterrows():
    row = np.concatenate([row.tolist()],axis=0)
    table.append(row)
    
x_data = np.concatenate([table],axis=0)

# COMMAND ----------

def partition_data(subjects, subj_inputs, x_data, y_data):
    """Retrieval of subject data based on subject indices passed in parameters.

    Args:
        subjects (list): List of subjects index.
        subj_inputs (List): List of index subject separation in input data.
        x_data (np.array): Input data
        y_data (np.array): Output data

    Returns:
        tuple: Partionned input data, Partionned output data
    """

    # subjects = tuple (0-based)
    x_part = None
    y_part = None
    for subj in subjects:
        skip = sum(subj_inputs[:subj])
        num = subj_inputs[subj]
        xx = x_data[skip : skip + num]
        yy = y_data[skip : skip + num]
        if x_part is None:
            x_part = xx.copy()
            y_part = yy.copy()
        else:
            x_part = np.vstack((x_part, xx))  # vstack creates a copy of the data
            y_part = np.vstack((y_part, yy)
    return x_part, y_part

# COMMAND ----------

def create_model(hyperparmeters, num_classes, num_features):
    """Create LSTM model.

    Args:
        args (argparse): Argparse arguments objects.
        num_classes (int): Number of classes.
        num_features (int): Number of input features

    Returns:
        tuple: Partionned input data, Partionned output data
    """
        
    model = Sequential()
    
    model.add(Input(shape=(hyperparmeters["window"], num_features)))
    model.add(Dense(hyperparmeters["num_cell_dense1"], name='dense1'))
    model.add(BatchNormalization(name='norm'))
    
    model.add(LSTM(hyperparmeters["num_cell_lstm1"], return_sequences=True, name='lstm1'))
    model.add(Dropout(hyperparmeters["dropout_rate"], name='drop2'))
    
    model.add(LSTM(hyperparmeters["num_cell_lstm2"], return_sequences=True, name='lstm2'))
    model.add(Dropout(hyperparmeters["dropout_rate"], name='drop3'))
    
    model.add(LSTM(hyperparmeters["num_cell_lstm3"], name='lstm3'))
    model.add(Dropout(0.5, name='drop4'))
    
    model.add(Dense(num_classes, name='dense2')) 
    
    optimizer = Adam(hyperparmeters["learning_rate"])
    
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer=optimizer, metrics=['accuracy'])
    return model


# COMMAND ----------

hyperparameters = {
    "epochs":1,
    "learning_rate":0.01,
    "train_subjects":(0,1,2,3),
    "validation_subjects": (4,),
    "test_subjects": (5,6),
    "num_cell_dense1":32,
    "num_cell_lstm1":32,
    "num_cell_lstm2":32,
    "num_cell_lstm3":32,
    "dropout_rate":0.2,
    "window":1200,
    "overlap":600
}

# COMMAND ----------

x_data_train, y_data_train = partition_data(hyperparameters["train_subjects"], subj_inputs, x_data, y_data)
x_data_validation, y_data_validation = partition_data(hyperparameters["validation_subjects"], subj_inputs, x_data, y_data)
x_data_test, y_data_test = partition_data(hyperparameters["test_subjects"], subj_inputs, x_data, y_data)

# COMMAND ----------

num_classes = len(set(y_data.flatten()))
num_features = x_data.shape[2]

with mlflow.start_run(run_name='lstm_har') as run:
    mlflow.log_param("epochs", hyperparameters["epochs"])
    mlflow.log_param("learning_rate", hyperparameters["learning_rate"])
    mlflow.log_param("train_subjects", hyperparameters["train_subjects"])
    mlflow.log_param("train_subjects", hyperparameters["validation_subjects"])
    mlflow.log_param("test_subjects", hyperparameters["test_subjects"])
    mlflow.log_param("num_cell_dense1", hyperparameters["num_cell_dense1"])
    mlflow.log_param("num_cell_lstm1", hyperparameters["num_cell_lstm1"])
    mlflow.log_param("num_cell_lstm2", hyperparameters["num_cell_lstm2"])
    mlflow.log_param("num_cell_lstm3", hyperparameters["num_cell_lstm3"])
    mlflow.log_param("dropout_rate", hyperparameters["dropout_rate"])
    mlflow.log_param("window", hyperparameters["window"])
    mlflow.log_param("overlap", hyperparameters["overlap"])
    
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
    
    signature = infer_signature(x_data_test, model.predict(x_data_test))
    print(signature)

    input_example = np.expand_dims(x_data_train[0], axis=0)
        
    mlflow.keras.log_model(model, "rnn-model", signature=signature, input_example=input_example, registered_model_name="rnn-model")
