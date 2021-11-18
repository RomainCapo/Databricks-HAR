# Databricks notebook source
# MAGIC %md
# MAGIC # Include tests functions

# COMMAND ----------

import unittest

class TestModel(unittest.TestCase):

    def test_shape_input(self):
        x_data_shape = x_data.shape[1:]
        excepted_shape = (window,num_features)
        self.assertEqual(x_data_shape, excepted_shape, f"Should be ({window}, {num_features})")

    def test_shape_output(self):
        y_data_shape = y_data.shape[1:]
        excepted_shape = (1,)
        self.assertEqual(y_data_shape, excepted_shape, "Should be (1,)")
        
    def test_model_input(self):
        model_input_shape = tuple(m.layers[0].input.get_shape().as_list())[1:]
        excepted_shape = (window,num_features)
        self.assertEqual(model_input_shape, excepted_shape, f"Should be ({window}, {num_features})")
        
    def test_model_output(self):
        model_output_shape = tuple(m.layers[-1].input.get_shape().as_list())[1:]
        excepted_shape = (num_classes,)
        self.assertEqual(model_output_shape, excepted_shape, "Should be (1,)")
        
    def test_prediction_shape(self):
        num_samples = 3
        x_test_sample = x_data_test[0:num_samples,:,:]
        y_pred_sample = model.predict(x_test_sample)
        model_pred_shape = y_pred_sample.shape
        excepted_shape = (num_samples, num_classes)
        self.assertEqual(model_pred_shape, excepted_shape, f"Should be ({num_samples}, {num_classes}")

# COMMAND ----------

def execute_test(test_names):
    test_result = []
    
    for test in test_names:
        try:
            suite = unittest.TestSuite()
            suite.addTest(TestModel(test))
            runner = unittest.TextTestRunner()
            runner.run(suite)
            mlflow.log_param(test, "Passed")
            test_result.append(True)
        except Exception as e:
            mlflow.log_param(test, "Failed")
            test_result.append(False)
            print(e)
    return test_result
