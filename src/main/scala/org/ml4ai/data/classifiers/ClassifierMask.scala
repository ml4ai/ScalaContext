package org.ml4ai.data.classifiers

import org.ml4ai.data.utils.AggegatedRow

trait ClassifierMask{
  def fit(xTrain: Seq[AggegatedRow]): Unit

  def predict(xTest: Seq[AggegatedRow]):Array[Int]
  def saveModel(fileName: String): Unit
  def loadFrom(fileName: String):LinearSVMWrapper
}
