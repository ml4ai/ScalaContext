package org.ml4ai.data

trait ClassifierMask {
  def fit(xTrain: Array[Array[Double]], yTrain: Seq[Boolean]):Array[Array[Double]]
  def predict(xTest: Array[Array[Double]]):Seq[Boolean]
}

