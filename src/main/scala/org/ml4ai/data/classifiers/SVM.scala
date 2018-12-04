package org.ml4ai.data.classifiers
import org.clulab.learning.{LibSVMClassifier, LinearKernel, PolynomialKernel, RBFKernel, RVFDataset, RVFDatum}
import org.clulab.struct.Counter
case class SVM(classifier: LibSVMClassifier[Int, Double]) extends ClassifierMask {
  override def fit(xTrain: Array[Array[Double]], yTrain: Array[Int]) :Unit = ()

  override def predict(xTest:Array[Array[Double]]): Array[Int] = List.fill(xTest.size)(1).toArray


  private def mkRVFDatum[L](label:L, features:Array[Double]):RVFDatum[L, Double] = {
    val c = new Counter[Double]
    for(f <- features) c.incrementCount(f)
    new RVFDatum[L, Double](label, c)
  }

  def mkRVFDataSet(labels: Array[Int], dataSet:Array[Array[Double]]):RVFDataset[Int, Double] = {
    val dataSetToReturn = new RVFDataset[Int, Double]()
    val tupIter = dataSet zip labels
    for((d,l) <- tupIter) {
      val currentDatum = mkRVFDatum(l,d)
      dataSetToReturn += currentDatum
    }
    dataSetToReturn
  }


}
