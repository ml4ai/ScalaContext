package org.ml4ai.data

case class Baseline(k:Int) extends ClassifierMask {
  override def fit(xTrain: Array[Array[Double]], yTrain: Array[Int]): Unit = {}

  override def predict(xTest: Array[Array[Double]]): Array[Int] = {
    val toPass = xTest.map(s => s(1))
    deterministicSentenceDist(toPass, k)
  }

  private def deterministicSentenceDist(sentDistVals:Array[Double], k:Int):Array[Int] = {
    val res = sentDistVals.map(s => if(s <= k) 1 else 0)
    res
  }
}

