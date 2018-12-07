package org.ml4ai

import java.util.zip._

import org.clulab.learning.{LibSVMClassifier, LinearKernel}

import scala.collection.mutable
import org.ml4ai.data.classifiers.{Baseline, DummyClassifier, SVM}
import org.ml4ai.data.utils.correctDataPrep.{AggregatedRowNew, Balancer, FoldMaker, Utils}

import scala.io.Source
object Main extends App {
  val (allFeatures,rows) = AggregatedRowNew.fromStream(new GZIPInputStream(getClass.getResourceAsStream("/grouped_features.csv.gz")))
  val rows2 = rows.filter(_.PMCID != "b'PMC4204162'")
  val bufferedFoldIndices = Source.fromFile("./src/main/resources/cv_folds_val_4.csv")
  val foldsFromCSV = FoldMaker.getFoldsPerPaper(bufferedFoldIndices)

  // baseline results
  var scoreDictionary = collection.mutable.Map[String, ((String, Double, Double, Double), (String, Double, Double, Double))]()
  val baselineResults = FoldMaker.baselineController(foldsFromCSV, rows2)
  scoreDictionary ++= baselineResults


  // SVM classifier
  val SVMClassifier = new LibSVMClassifier[Int, String](LinearKernel)
  val svmInstance = new SVM(SVMClassifier)
  val giantTruthTestLabel = new mutable.ArrayBuffer[Int]()
  val giantPredTestLabel = new mutable.ArrayBuffer[Int]()
  val giantTruthValLabel = new mutable.ArrayBuffer[Int]()
  val giantPredValLabel = new mutable.ArrayBuffer[Int]()
  for((train,validate,test) <- foldsFromCSV) {
    val trainingData = train.collect{case x:Int => rows2(x)}
    val balancedTrainingData = Balancer.balanceByPaperAgg(trainingData, 1)

    val trainingLabels = DummyClassifier.convertOptionalToBool(balancedTrainingData)
    val labelsToInt = DummyClassifier.convertBooleansToInt(trainingLabels)
    giantTruthTestLabel++=labelsToInt

    val tups = svmInstance.constructTupsForRVF(balancedTrainingData)
    val (trainDataSet, _) = svmInstance.mkRVFDataSet(labelsToInt,tups)
    svmInstance.fit(trainDataSet)


    val validationData = validate.collect{case v:Int => rows2(v)}
    val validationLabels = DummyClassifier.convertOptionalToBool(validationData)
    val valLabelsTruth = DummyClassifier.convertBooleansToInt(validationLabels)
    giantTruthValLabel ++= valLabelsTruth
    val tupsVal = svmInstance.constructTupsForRVF(validationData)
    val (_, valDatumCollect) = svmInstance.mkRVFDataSet(valLabelsTruth, tupsVal)
    val valLabelsPred = valDatumCollect.map(vd => svmInstance.predict(vd))
    giantPredValLabel ++= valLabelsPred


    val testingData = test.collect{case t: Int => rows2(t)}
    val testLabels = DummyClassifier.convertOptionalToBool(testingData)
    val testLabelsTruth = DummyClassifier.convertBooleansToInt(testLabels)
    giantTruthTestLabel ++= testLabelsTruth
    val tupsTruth = svmInstance.constructTupsForRVF(testingData)
    val (_, testDatumCollect) = svmInstance.mkRVFDataSet(testLabelsTruth, tupsTruth)
    val testLabelsPred = testDatumCollect.map(td => svmInstance.predict(td))
    giantPredTestLabel ++= testLabelsPred
  }
  val svmScore = svmInstance.scoreMaker("Linear SVM", giantTruthValLabel.toArray, giantPredValLabel.toArray, giantTruthTestLabel.toArray, giantPredTestLabel.toArray)
  scoreDictionary ++= svmScore
  println(scoreDictionary)
}
