package org.ml4ai

import com.typesafe.config.ConfigFactory
import org.clulab.learning.LinearSVMClassifier
import org.ml4ai.data.classifiers.LinearSVMWrapper
import org.ml4ai.data.utils.{FoldMaker, CodeUtils, AggregatedRow}
import scala.io.Source

object ML4AIPackageLauncher extends App {

  // *** DATA LOADING *****===========
  val config = ConfigFactory.load()
  val foldsPath = config.getString("folds.filePath")
  val allFeatsPath = config.getString("features.allFeatures")
  val groupedFeaturesPath = config.getString("features.groupedFeatures")
  val hardCodedFeaturePath = config.getString("features.hardCodedFeatures")
  val hardCodedInputRowFeatures = config.getString("features.hardCodedInputRowFeatures")
  val (allFeatures,rows) = AggregatedRow.fromFile(groupedFeaturesPath)
  CodeUtils.writeAllFeaturesToFile(allFeatures, allFeatsPath)
  CodeUtils.writeHardcodedFeaturesToFile(hardCodedFeaturePath)
  CodeUtils.writeInputRowFeatures(hardCodedInputRowFeatures)
  val rows2 = rows.filter(_.PMCID != "b'PMC4204162'")
  val bufferedFoldIndices = Source.fromFile(foldsPath)
  val foldsFromCSV = FoldMaker.getFoldsPerPaper(bufferedFoldIndices)
  val trainValCombined = CodeUtils.combineTrainVal(foldsFromCSV)

  // =========================== BASELINE RESULTS ===========================
  // baseline results
  // The F1 score seems to have changed, perform debugging. It should be 0.5333
  var scoreDictionary = collection.mutable.Map[String, (String, Double, Double, Double)]()
  /*val (truthTest, predTest) = FoldMaker.baselineController(foldsFromCSV, rows2)
  val baselineResults = Utils.scoreMaker("baseline", truthTest, predTest)
  scoreDictionary ++= baselineResults*/
  //========================== CONCLUDING BASELINE RESULTS ==========================


  // =========================== LINEAR SVM RESULTS ===========================
  val fileName = config.getString("svm.untrainedModel")

  // svm instance using liblinear
  val SVMClassifier = new LinearSVMClassifier[Int, String](C = 0.001, eps = 0.001, bias = false)
  val svmInstance = new LinearSVMWrapper(SVMClassifier)
  svmInstance.saveModel(fileName)
  val loadedModelWrapper = svmInstance.loadFrom(fileName)
  // the loadedModel variable is an instance of LinearSVMWrapper. If you want access to the LinearSVMClassifier instance,
  // just call loadedModel.classifier.
  val (truthTestSVM, predTestSVM) = FoldMaker.svmControllerLinearSVM(loadedModelWrapper, trainValCombined, rows2)
  val svmResult = CodeUtils.scoreMaker("Linear SVM", truthTestSVM, predTestSVM)
  scoreDictionary ++= svmResult
  //========================== CONCLUDING LINEAR SVM RESULTS ==========================

  println("size of score dictionary: " + scoreDictionary.size)
  println(scoreDictionary)


}
