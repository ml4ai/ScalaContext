package org.ml4ai.data
object Classifier extends ClassifierMask {
  override def fit(xTrain: Array[Array[Double]], yTrain: Seq[Boolean]): Array[Array[Double]] = xTrain

  override def predict(xTest:Array[Array[Double]]): Seq[Boolean] = List.fill(xTest.size)(true)
  def convertBooleansToInt(labels: Seq[Boolean]):Array[Int] = {

    val toReturn = labels.map(l => l match {
      case true => 1
      case false => 0
    })
    toReturn.toArray
  }

  def convertOptionalToBool(rows: Seq[AggregatedRow]): Seq[Boolean] = {
    rows.map(x => x.label match {
      case Some(x) => x
      case _ => false
    })
  }
}