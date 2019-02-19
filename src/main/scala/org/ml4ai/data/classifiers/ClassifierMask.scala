package org.ml4ai.data.classifiers



import org.clulab.learning.{LinearSVMClassifier}


trait ClassifierMask{
  def fit(xTrain: Array[Array[Double]], yTrain: Array[Int])
  def predict(xTest: Array[Array[Double]]):Array[Int]
  def scoreMaker(name: String, truthTest:Array[Int], predTest:Array[Int]):Map[String, (String, Double, Double, Double)]
  def saveModel(fileName: String): Unit
  def loadFrom(fileName: String):LinearSVMWrapper
}
