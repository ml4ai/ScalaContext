package org.ml4ai.data.classifiers

import org.ml4ai.data.utils.correctDataPrep.{AggregatedRowNew}
import smile.classification._
case class GradTreeBoost(xTrain:Array[Array[Double]], yTrain:Array[Int], nEst:Int = 300) extends ClassifierMask {
  private val gradBoostInstance = gbm(xTrain, yTrain, null, nEst, shrinkage = 0.1, subsample = 1.0)
  override def predict(xTest: Seq[AggregatedRowNew]): Array[Int] = {
    val convert = dataConverter(xTest)
    convert.map(gradBoostInstance.predict(_))}

  override def fit(xTrain: Seq[AggregatedRowNew]): Unit = ()

  def fit(): GradientTreeBoost = {
    gradBoostInstance
  }

  override def saveModel(fileName: String): Unit = ()

  override def loadFrom(fileName: String): LinearSVMWrapper = null

  def dataConverter(data:Seq[AggregatedRowNew]):Array[Array[Double]] = {
    // sentence distance is at index 18 in the row. We will extract the value at this index, create an array from it,
    // and then add it to the resulting value.
    val sentDistIndex = 18
    val result = collection.mutable.ListBuffer[Array[Double]]()
    data.map(d => {
      result += Array(d.featureGroups(sentDistIndex))
    })
    result.toArray
  }
}
