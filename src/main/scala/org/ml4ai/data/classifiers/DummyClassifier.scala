package org.ml4ai.data.classifiers

import org.ml4ai.data.utils.correctDataPrep.AggregatedRowNew

object DummyClassifier extends ClassifierMask {
  override def fit(xTrain: Array[Array[Double]], yTrain: Array[Int]) :Unit = ()

  override def predict(xTest:Array[Array[Double]]): Array[Int] = List.fill(xTest.size)(1).toArray

  override def scoreMaker(name: String, truthTest:Array[Int], predTest:Array[Int]): Map[String,  (String, Double, Double, Double)] = {
    Map("dummy" -> ("test", 0.0,0.0,0.0))
  }
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
}
