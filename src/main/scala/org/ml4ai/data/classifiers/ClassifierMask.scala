package org.ml4ai.data.classifiers

import org.ml4ai.data.utils.AggregatedRow

trait ClassifierMask{
  def fit(xTrain: Seq[AggregatedRow]): Unit

  def predict(xTest: Seq[AggregatedRow]):Array[Int]
  def saveModel(fileName: String): Unit
  def loadFrom(fileName: String):LinearSVMWrapper
}
