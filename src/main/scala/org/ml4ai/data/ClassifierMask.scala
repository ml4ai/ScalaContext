package org.ml4ai.data

trait ClassifierMask {
  def fit(xTrain: Seq[AggregatedRow], yTrain: Seq[Boolean]):Seq[AggregatedRow]
  def predict(xTest: Seq[AggregatedRow]):Seq[Boolean]
}

