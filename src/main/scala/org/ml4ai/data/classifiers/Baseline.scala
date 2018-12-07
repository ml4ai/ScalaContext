package org.ml4ai.data.classifiers

case class Baseline(k:Int) extends ClassifierMask {
  override def fit(xTrain: Array[Array[Double]], yTrain: Array[Int]): Unit = {}

  override def predict(xTest: Array[Array[Double]]): Array[Int] = {
    val toPass = xTest.map(s => s(0))
    deterministicSentenceDist(toPass, k)
  }

  override def scoreMaker(name: String, truth: Array[Int], predicted: Array[Int], truthTest:Array[Int], predTest:Array[Int]): Map[String, ((String, Double, Double, Double), (String, Double, Double, Double))] = {
    Map("baseline" -> (("validation", 0.0, 0.0,0.0), ("test", 0.0,0.0,0.0)))
  }

  private def deterministicSentenceDist(sentDistVals:Array[Double], k:Int):Array[Int] = {
    val res = sentDistVals.map(s => if(s <= k) 1 else 0)
    res
  }
}
