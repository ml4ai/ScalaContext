package org.ml4ai.data.classifiers
import java.io.Writer

import org.clulab.learning.LinearSVMClassifier
import org.ml4ai.data.utils.correctDataPrep.Utils
import smile.classification._
case class GradTreeBoost(xTrain:Array[Array[Double]], yTrain:Array[Int], nEst:Int = 300) extends ClassifierMask {
  private val gradBoostInstance = gbm(xTrain, yTrain, null, nEst, shrinkage = 0.1, subsample = 1.0)
  override def predict(xTest: Array[Array[Double]]): Array[Int] = xTest.map(gradBoostInstance.predict(_))

  override def fit(xTrain: Array[Array[Double]], yTrain: Array[Int]): Unit = ()

  def fit(): GradientTreeBoost = {
    gradBoostInstance
  }

  override def saveModel(fileName: String): Unit = ()

  override def loadFrom(fileName: String): LinearSVMWrapper = null


}
