package org.ml4ai.data.classifiers
import org.ml4ai.data.utils.correctDataPrep.Utils
import smile.classification._
case class GradTreeBoost(xTrain:Array[Array[Double]], yTrain:Array[Int], nEst:Int = 300) extends ClassifierMask {
  private val gradBoostInstance = gbm(xTrain, yTrain, null, nEst)
  override def predict(xTest: Array[Array[Double]]): Array[Int] = xTest.map(gradBoostInstance.predict(_))

  override def scoreMaker(name: String, truthTest: Array[Int], predTest: Array[Int]): Map[String, (String, Double, Double, Double)] = {
    val countsTest = Utils.predictCounts(truthTest, predTest)
    val precTest = Utils.precision(countsTest)
    val recallTest = Utils.recall(countsTest)
    val f1Test = Utils.f1(countsTest)
    val testTup = ("test", precTest, recallTest, f1Test)
    val mapToReturn = Map(name -> testTup)
    mapToReturn
  }

  override def train(xTrain: Array[Array[Double]], yTrain: Array[Int]): Unit = ()

  def train(): GradientTreeBoost = {
    gradBoostInstance
  }


}
