package org.ml4ai.data.classifiers

case class Baseline(k:Int) extends ClassifierMask {
  override def fit(xTrain: Array[Array[Double]], yTrain: Array[Int]): Unit = {}

  override def predict(xTest: Array[Array[Double]]): Array[Int] = {
    val toPass = xTest.map(s => s(0))
    deterministicSentenceDist(toPass, k)
  }

  private def deterministicSentenceDist(sentDistVals:Array[Double], k:Int):Array[Int] = {
    val res = sentDistVals.map(s => if(s <= k) 1 else 0)
    res
  }

  override def saveModel(fileName: String): Unit = ()

  override def loadFrom(fileName: String): LinearSVMWrapper = null
}
