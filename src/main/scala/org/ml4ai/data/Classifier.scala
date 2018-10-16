package org.ml4ai.data
import scala.collection.mutable
object Classifier extends ClassifierMask {
  override def fit(xTrain: Seq[AggregatedRow], yTrain: Seq[Boolean]): Seq[AggregatedRow] = xTrain

  override def predict(xTest: Seq[AggregatedRow]): Seq[Boolean] = Seq(true)

  def convertBooleansToInt(labels: Seq[Boolean]):Array[Int] = {

    val toReturn = labels.map(l => l match {
      case true => 1
      case false => 0
      case _ => 0
    })
    toReturn.toArray
  }
}