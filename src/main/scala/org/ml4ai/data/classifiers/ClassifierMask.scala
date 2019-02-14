package org.ml4ai.data.classifiers

trait ClassifierMask{
  def fit(xTrain: Array[Array[Double]], yTrain: Array[Int])
  def predict(xTest: Array[Array[Double]]):Array[Int]
  def scoreMaker(name: String, truthTest:Array[Int], predTest:Array[Int]):Map[String, (String, Double, Double, Double)]
}
