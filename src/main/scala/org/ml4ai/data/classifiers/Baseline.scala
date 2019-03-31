package org.ml4ai.data.classifiers

import org.ml4ai.data.utils.AggregatedRow

case class Baseline(k:Int) extends ClassifierMask {
  override def fit(xTrain: Seq[AggregatedRow]): Unit = {}

  override def predict(xTest: Seq[AggregatedRow]): Array[Int] = {
    val convert = dataConverter(xTest)
    val toPass = convert.map(s => s(0))
    deterministicSentenceDist(toPass, k)
  }

  private def deterministicSentenceDist(sentDistVals:Array[Double], k:Int):Array[Int] = {
    val res = sentDistVals.map(s => if(s <= k) 1 else 0)
    res
  }

  override def saveModel(fileName: String): Unit = ()

  override def loadFrom(fileName: String): LinearSVMWrapper = null

  def dataConverter(data:Seq[AggregatedRow]):Array[Array[Double]] = {
    // sentence distance is at index 18 in the row. We will extract the value at this index, create an array from it,
    // and then add it to the resulting value.
    val sentDistIndex = 18
    val result = collection.mutable.ListBuffer[Array[Double]]()
    data.map(d => {
      result += Array(d.featureGroups(sentDistIndex))
    })
    result.toArray
  }

  def createLabels(data:Seq[AggregatedRow]):Array[Int] = {
    val currentTruthTest = DummyClassifier.convertOptionalToBool(data)
    val currentTruthTestInt = DummyClassifier.convertBooleansToInt(currentTruthTest)
    currentTruthTestInt
  }
}
