from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class TitanicTensorExample:
  def __init__(self, ):
    # Load dataset.
    self.dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
    self.dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
    self.y_train = self.dftrain.pop('survived')
    self.y_eval = self.dfeval.pop('survived')

    self.CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
    self.NUMERIC_COLUMNS = ['age', 'fare']


    self.feature_columns = self.get_feature_colums()

    self.train_input_fn = self.make_input_fn(self.dftrain, self.y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
    self.eval_input_fn = self.make_input_fn(self.dfeval, self.y_eval, num_epochs=1, shuffle=False)

    self.linear_est = None
    self.linear_est_result = None
  


  def get_feature_colums(self):
    feature_columns = []
    for feature_name in self.CATEGORICAL_COLUMNS:
      vocabulary = self.dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
      feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

    for feature_name in self.NUMERIC_COLUMNS:
      feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
    
    return feature_columns


  def make_input_fn(self, data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():  # inner function, this will be returned
      ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
      if shuffle:
        ds = ds.shuffle(1000)  # randomize order of data
      ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
      return ds  # return a batch of the dataset
    return input_function  # return a function object for 
    

  def perform_linear_estimate(self):
    linear_est = tf.estimator.LinearClassifier(feature_columns=self.feature_columns)

    linear_est.train(self.train_input_fn)  # train
    result = linear_est.evaluate(self.eval_input_fn)  # get model metrics/stats by testing on tetsing data
    self.linear_est = linear_est
    self.linear_est_result = result
    print(f"Finished training model, based on testing data, there is a {str(round(result['accuracy'], 2))}% accuracy rate")

  def predict_input_fn(self, features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

  def get_prediction_object(self):
    features_string = ['sex', 'class', 'deck', 'embark_town', 'alone']
    features_float = ['age', 'fare']
    features_int = ['n_siblings_spouses', 'parch']
    predict = {}

    print("Please type string values as prompted.")
    for feature in features_string:
      valid = True
      while valid: 
        val = str(input(feature + ": "))
        if type(val) == str: valid = False

      predict[feature] = [val]

    print("Please type float values as prompted.")
    for float_feature in features_float:
      valid = True
      while valid: 
        val = float(input(float_feature + ": "))
        if type(val) == float: valid = False

      predict[float_feature] = [val]

    print("Please type int values as prompted.")
    for int_feature in features_int:
      valid = True
      while valid: 
        val = int(input(int_feature + ": "))
        if type(val) == int: valid = False

      predict[int_feature] = [val]

    return predict
  
  def perform_prediction(self, prediction_object):
    predictions = self.linear_est.predict(input_fn=lambda: self.predict_input_fn(prediction_object))
    for pred_dict in predictions:
      # print(pred_dict)
      class_id = pred_dict['class_ids'][0]
      probability = pred_dict['probabilities'][class_id]
      survival_chance = "{:.2%}".format(probability)
      survived = class_id == 1
      # print(class_id)
      result = f"Based on the input provided, we could expect this individual to have a {survival_chance} chance of surviving."
      print(result)
      return result





if __name__ == "__main__":
  titanic_nn = TitanicTensorExample()
  titanic_nn.perform_linear_estimate()

  while True:
    print("Starting a new prediction")
    prediction_object = titanic_nn.get_prediction_object()
    prediction_result = titanic_nn.perform_prediction(prediction_object)