package org.ml4ai.data.classifiers

import java.io.Writer

import org.clulab.learning.LinearSVMClassifier
import org.ml4ai.data.utils.correctDataPrep.AggregatedRowNew

object DummyClassifier extends ClassifierMask {
  override def fit(xTrain: Array[Array[Double]], yTrain: Array[Int]) :Unit = ()

  override def predict(xTest:Array[Array[Double]]): Array[Int] = List.fill(xTest.size)(1).toArray

  def convertBooleansToInt(labels: Seq[Boolean]):Array[Int] = {

    val toReturn = labels.map(l => l match {
      case true => 1
      case false => 0
    })
    toReturn.toArray
  }

  def convertOptionalToBool(rows: Seq[AggregatedRowNew]): Seq[Boolean] = {
    rows.map(x => x.label match {
      case Some(x) => x
      case _ => false
    })
  }

  override def saveModel(fileName: String): Unit = ()

  override def loadFrom(fileName: String): LinearSVMWrapper = null
}
