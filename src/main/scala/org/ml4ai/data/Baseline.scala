package org.ml4ai.data
import scala.collection.mutable
case class Baseline(k:Int, sentDistIndex:Array[Int]) extends ClassifierMask {
  override def fit(xTrain: Array[Array[Double]], yTrain: Array[Int]): Unit = {}

  override def predict(xTest: Array[Array[Double]]): Array[Int] = {
    val zip = sentDistIndex.zipWithIndex
    val distVals = new mutable.ListBuffer[Double]
    for((z,ix) <- zip) {
      val ar = xTest(ix)
      val value = ar(z)
      distVals += value
    }
    deterministicSentenceDist(distVals.toArray, k)
  }

  private def deterministicSentenceDist(input: Array[Double], k:Int): Array[Int] = {
    input.map(i => if(i <= k) 1 else 0)
  }
}

