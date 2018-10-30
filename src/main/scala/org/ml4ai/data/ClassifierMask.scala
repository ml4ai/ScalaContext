package org.ml4ai.data

trait ClassifierMask{
  def fit(xTrain: Array[Array[Double]], yTrain: Array[Int])
  def predict(xTest: Array[Array[Double]]):Array[Int]
}

