package org.ml4ai.data.classifiers
import org.ml4ai.data.utils.correctDataPrep.AggregatedRowNew

object DummyClassifier extends ClassifierMask {
  override def fit(xTrain: Seq[AggregatedRowNew]):Unit = ()

  override def predict(xTest: Seq[AggregatedRowNew]): Array[Int] = List.fill(xTest.size)(1).toArray

  def convertBooleansToInt(labels: Seq[Boolean]):Array[Int] = {

    val toReturn = labels.map {
      case true => 1
      case false => 0
    }
    toReturn.toArray
  }

  def convertOptionalToBool(rows: Seq[AggregatedRowNew]): Seq[Boolean] = {
    rows.map(_.label match {
      case Some(x) => x
      case _ => false
    })
  }

  override def saveModel(fileName: String): Unit = ()

  override def loadFrom(fileName: String): ClassifierMask = throw new NotImplementedError()
}
