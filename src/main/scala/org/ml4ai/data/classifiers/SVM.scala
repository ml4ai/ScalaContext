package org.ml4ai.data.classifiers
import org.clulab.learning.{LibSVMClassifier, LinearKernel, PolynomialKernel, RBFKernel, RVFDataset, RVFDatum}
import org.clulab.struct.Counter
import org.ml4ai.data.utils.correctDataPrep.{AggregatedRowNew, Utils}
case class SVM(classifier: LibSVMClassifier[Int, String]) extends ClassifierMask {
  override def fit(xTrain: Array[Array[Double]], yTrain: Array[Int]) :Unit = ()

  def fit(xTrain: RVFDataset[Int, String]):Unit = classifier.train(xTrain)

  override def predict(xTest:Array[Array[Double]]): Array[Int] = List.fill(xTest.size)(1).toArray

  def predict(testDatum:RVFDatum[Int, String]):Int = {
    classifier.classOf(testDatum)
  }

  override def scoreMaker(name: String, truthVal: Array[Int], predVal: Array[Int], truthTest:Array[Int], predTest:Array[Int]): Map[String, ((String, Double, Double, Double), (String, Double, Double, Double))] = {
    val countsVal = Utils.predictCounts(truthVal, predVal)
    val precVal = Utils.precision(countsVal)
    val recallVal = Utils.recall(countsVal)
    val f1Val = Utils.f1(countsVal)
    val valTup = ("validation", precVal, recallVal, f1Val)

    val countsTest = Utils.predictCounts(truthTest, predTest)
    val precTest = Utils.precision(countsTest)
    val recallTest = Utils.recall(countsTest)
    val f1Test = Utils.f1(countsTest)
    val testTup = ("test", precTest, recallTest, f1Test)
    Map(name -> (valTup, testTup))
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
