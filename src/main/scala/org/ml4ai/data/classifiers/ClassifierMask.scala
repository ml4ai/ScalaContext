package org.ml4ai.data.classifiers

import org.ml4ai.data.utils.correctDataPrep.AggregatedRowNew

trait ClassifierMask{
  def fit(xTrain: Seq[AggregatedRowNew]): Unit

  def predict(xTest: Seq[AggregatedRowNew]):Array[Int]
  def saveModel(fileName: String): Unit
  def loadFrom(fileName: String):LinearSVMWrapper
}
