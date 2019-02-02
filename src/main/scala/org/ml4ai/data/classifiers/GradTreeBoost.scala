package org.ml4ai.data.classifiers
import smile.classification._
case class GradTreeBoost(xTrain:Array[Array[Double]], yTrain:Array[Int], nEst:Int = 300) extends ClassifierMask {
  override def predict(xTest: Array[Array[Double]]): Array[Int] = ???

  override def scoreMaker(name: String, truthTest: Array[Int], predTest: Array[Int]): Map[String, (String, Double, Double, Double)] = ???

  override def train(xTrain: Array[Array[Double]], yTrain: Array[Int]): Unit = ???

}
