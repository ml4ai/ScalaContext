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
    val trainVal = train.toSet ++ validate.toSet
    val toAdd = (trainVal.toArray, test)
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


  // SVM classifier
  //val SVMClassifier = new LibSVMClassifier[Int, String](LinearKernel, C= 0.001, eps = 0.001)
  val SVMClassifier = new LinearSVMClassifier[Int, String](C = 0.001, eps = 0.001, bias = false)
  //val svmInstance = new SVM(SVMClassifier)
  val svmInstance = new LinearSVMWrapper(SVMClassifier)
  val giantTruthTestLabel = new mutable.ArrayBuffer[Int]()
  val giantPredTestLabel = new mutable.ArrayBuffer[Int]()
  val giantTruthValLabel = new mutable.ArrayBuffer[Int]()
  val giantPredValLabel = new mutable.ArrayBuffer[Int]()
  for((train,test) <- trainValCombined.toArray) {
    val trainingData = train.collect{case x:Int => rows2(x)}
    val balancedTrainingData = Balancer.balanceByPaperAgg(trainingData, 1)

    val trainingLabels = DummyClassifier.convertOptionalToBool(balancedTrainingData)
    val labelsToInt = DummyClassifier.convertBooleansToInt(trainingLabels)

    val tups = svmInstance.constructTupsForRVF(balancedTrainingData)
    val (trainDataSet, _) = svmInstance.mkRVFDataSet(labelsToInt,tups)
    svmInstance.train(trainDataSet)


    val testingData = test.collect{case t: Int => rows2(t)}
    val testLabels = DummyClassifier.convertOptionalToBool(testingData)
    val testLabelsTruth = DummyClassifier.convertBooleansToInt(testLabels)
    giantTruthTestLabel ++= testLabelsTruth
    val tupsTruth = svmInstance.constructTupsForRVF(testingData)
    val (_, testDatumCollect) = svmInstance.mkRVFDataSet(testLabelsTruth, tupsTruth)
    val testLabelsPred = testDatumCollect.map(td => svmInstance.predict(td))
    giantPredTestLabel ++= testLabelsPred
  }

  val svmScore = svmInstance.scoreMaker("Linear SVM",  giantTruthTestLabel.toArray, giantPredTestLabel.toArray)
  scoreDictionary ++= svmScore
  println("size of score dictionary: " + scoreDictionary.size)
  println(scoreDictionary)
}
