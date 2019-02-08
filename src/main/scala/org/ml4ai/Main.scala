package org.ml4ai

import java.util.zip._

import org.clulab.learning.{LibSVMClassifier, LinearKernel, LinearSVMClassifier}

import scala.collection.mutable
import org.ml4ai.data.classifiers.{DummyClassifier, LinearSVMWrapper, SVM}
import org.ml4ai.data.utils.correctDataPrep.{AggregatedRowNew, Balancer, FoldMaker, Utils}
//next steps: try LibLinear classifier, and make sure RVFDataSet allots accurate values to its features
import scala.io.Source
object Main extends App {
  val (allFeatures,rows) = AggregatedRowNew.fromStream(new GZIPInputStream(getClass.getResourceAsStream("/grouped_features.csv.gz")))
  val rows2 = rows.filter(_.PMCID != "b'PMC4204162'")
  val bufferedFoldIndices = Source.fromFile("./src/main/resources/cv_folds_val_4.csv")
  val foldsFromCSV = FoldMaker.getFoldsPerPaper(bufferedFoldIndices)
  val trainValCombined = collection.mutable.ListBuffer[(Array[Int], Array[Int])]()
  for((train,validate,test) <- foldsFromCSV) {
    val trainVal = train ++ validate
    val toAdd = (trainVal, test)
    trainValCombined += toAdd
  }


  // =========================== BASELINE RESULTS ===========================
  // baseline results
  var scoreDictionary = collection.mutable.Map[String, (String, Double, Double, Double)]()
  val (truthTest, predTest) = FoldMaker.baselineController(foldsFromCSV, rows2)
  val countsTest = Utils.predictCounts(truthTest, predTest)
  val precTest = Utils.precision(countsTest)
  val recallTest = Utils.recall(countsTest)
  val f1Test = Utils.f1(countsTest)
  val testTup = ("test", precTest, recallTest, f1Test)
  val baselineResults = Map("baseline" -> testTup)
  scoreDictionary ++= baselineResults
  //========================== CONCLUDING BASELINE RESULTS ==========================


  // =========================== LINEAR SVM RESULTS ===========================
  // svm instance using LibSVM with linear kernel
  //val SVMClassifier = new LibSVMClassifier[Int, String](LinearKernel, C= 0.001, eps = 0.001)
  //val svmInstance = new SVM(SVMClassifier)

  // svm instance using liblinear
  val SVMClassifier = new LinearSVMClassifier[Int, String](C = 0.001, eps = 0.001, bias = false)
  val svmInstance = new LinearSVMWrapper(SVMClassifier)

  val (truthTestSVM, predTestSVM) = FoldMaker.svmController(svmInstance, trainValCombined.toArray, rows2)
  val svmResult = svmInstance.scoreMaker("Linear SVM", truthTestSVM, predTestSVM)
  scoreDictionary ++= svmResult
  //========================== CONCLUDING LINEAR SVM RESULTS ==========================


  //=========================== GRADIENT TREE BOOST RESULTS ===========================
  val (truthTestGBT, predTestGBT) = FoldMaker.gradBoostController(trainValCombined.toArray, rows2)
  val gbtResult = FoldMaker.gbmScoreMaker("Gradient Tree Boost", truthTestGBT, predTestGBT)
  scoreDictionary ++= gbtResult
  //========================== CONCLUDING GRADIENT TREE BOOST RESULTS ==========================

  println("size of score dictionary: " + scoreDictionary.size)
  println(scoreDictionary)


}
