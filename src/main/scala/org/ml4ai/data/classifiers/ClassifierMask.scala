package org.ml4ai.data.classifiers

trait ClassifierMask{
  def fit(xTrain: Array[Array[Double]], yTrain: Array[Int])
  def predict(xTest: Array[Array[Double]]):Array[Int]
  def saveModel(fileName: String): Unit
  def loadFrom(fileName: String):LinearSVMWrapper
}
