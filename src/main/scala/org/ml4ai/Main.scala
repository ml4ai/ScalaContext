package org.ml4ai

import java.util.zip._

import org.clulab.learning.{LinearSVMClassifier}
import org.ml4ai.data.classifiers.LinearSVMWrapper
import org.ml4ai.data.utils.correctDataPrep.{AggregatedRowNew, FoldMaker, Utils}

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
  val baselineResults = Utils.scoreMaker("baseline", truthTest, predTest)
  scoreDictionary ++= baselineResults
  //========================== CONCLUDING BASELINE RESULTS ==========================


  // =========================== LINEAR SVM RESULTS ===========================
  val fileName = "./src/main/resources/svmModel.dat"

  // svm instance using liblinear
  val SVMClassifier = new LinearSVMClassifier[Int, String](C = 0.001, eps = 0.001, bias = false)
  val svmInstance = new LinearSVMWrapper(SVMClassifier)
  svmInstance.saveModel(fileName)
  val loadedModel = svmInstance.loadFrom(fileName)
  // the loadedModel variable is an instance of LinearSVMWrapper. If you want access to the LinearSVMClassifier instance,
  // just call loadedModel.classifier.

  //val (truthTestSVM, predTestSVM) = FoldMaker.svmControllerLinearSVM(svmInstance, trainValCombined.toArray, rows2)
  val (truthTestSVM, predTestSVM) = FoldMaker.svmControllerLinearSVM(loadedModel, trainValCombined.toArray, rows2)
  val svmResult = Utils.scoreMaker("Linear SVM", truthTestSVM, predTestSVM)
  scoreDictionary ++= svmResult
  //========================== CONCLUDING LINEAR SVM RESULTS ==========================


  println("size of score dictionary: " + scoreDictionary.size)
  println(scoreDictionary)


}
