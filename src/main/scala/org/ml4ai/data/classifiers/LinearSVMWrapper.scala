package org.ml4ai.data.classifiers
import java.io._

import org.clulab.struct.Counter
import org.ml4ai.data.utils.correctDataPrep.{AggregatedRowNew, Utils}
import org.clulab.learning._
case class LinearSVMWrapper(classifier: LinearSVMClassifier[Int,String]) extends ClassifierMask {
  override def fit(xTrain: Array[Array[Double]], yTrain: Array[Int]): Unit = ()

  def fit(xTrain: RVFDataset[Int, String]):Unit = classifier.train(xTrain)

  override def predict(xTest: Array[Array[Double]]): Array[Int] = List.fill(xTest.size)(1).toArray

  def predict(testDatum:RVFDatum[Int, String]):Int = {
    classifier.classOf(testDatum)
  }

  override def saveModel(fileName: String): Unit = {
    val os = new ObjectOutputStream(new FileOutputStream(fileName))
    os.writeObject(this)
    os.close()
  }

  override def loadFrom(fileName: String): LinearSVMWrapper = {
    val is = new ObjectInputStream(new FileInputStream(fileName))
    val c = is.readObject().asInstanceOf[LinearSVMWrapper]
    is.close()
    c
  }


  // Consider features as pairs of (feature name, feature value)
  private def mkRVFDatum[L](label:L, features:Array[(String, Double)]):RVFDatum[L, String] = {
    // In here, Counter[T] basically works as a dictionary, and String should be the simplest way to implement it
    // when you call c.incrementCount, you basically assign the feature called "featureName", the value in the second parameter ("inc")
    val c = new Counter[String]
    // In this loop we go through all the elements in features and initialize the counter with the values. It's weird but that's the way it was written
    for((featureName, featureValue) <- features) c.incrementCount(featureName, inc = featureValue)
    // Just changed the second type argument to string here. Label is the class, so, L can be Int to reflext 1 or 0
    new RVFDatum[L, String](label, c)
  }

  // Here I made the changes to reflect my comments above.
  def mkRVFDataSet(labels: Array[Int], dataSet:Array[Array[(String, Double)]]):(RVFDataset[Int, String], Array[RVFDatum[Int, String]]) = {
    val dataSetToReturn = new RVFDataset[Int, String]()
    val datumCollect = collection.mutable.ListBuffer[RVFDatum[Int, String]]()
    val tupIter = dataSet zip labels
    for((d,l) <- tupIter) {
      val currentDatum = mkRVFDatum(l,d)
      dataSetToReturn += currentDatum
      datumCollect += currentDatum
    }
    (dataSetToReturn, datumCollect.toArray)
  }

  def constructTupsForRVF(rows: Seq[AggregatedRowNew]):Array[Array[(String, Double)]] = {
    val toReturn = collection.mutable.ListBuffer[Array[(String,Double)]]()
    rows.map(r => {
      val featureVals = r.featureGroups
      val featureName = r.featureGroupNames
      val zipped = featureName zip featureVals
      toReturn += zipped
    })
    toReturn.toArray
  }


}
