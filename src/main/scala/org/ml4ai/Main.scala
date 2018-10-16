package org.ml4ai

import java.util.zip.GZIPInputStream


import data.InputRow
import data.Balancer
import data.AggregatedRow
import data.FoldMaker
import data.Classifier
import scala.collection.mutable
import smile.validation._
object Main extends App {

  val rows = InputRow.fromStream(new GZIPInputStream(getClass.getResourceAsStream("/features.csv.gz")))
  val balancedRows = Balancer.balanceByPaper(rows, 1)
  val aggregatedRows = AggregatedRow.fromRows(balancedRows)
  val folds = FoldMaker.getFolds(aggregatedRows)
  val dataOnly = aggregatedRows.values.toSeq
  val giantTruthTestLabel = new mutable.ArrayBuffer[Boolean]()
  val giantPredTestLabel = new mutable.ArrayBuffer[Boolean]()
  val giantTruthValLabel = new mutable.ArrayBuffer[Boolean]()
  val giantPredValLabel = new mutable.ArrayBuffer[Boolean]()
  for((train,validate,test) <- folds) {
      val trainingData = train.collect{case x:Int => dataOnly(x)}
      val validationData = validate.collect{case x:Int => dataOnly(x)}
      val testingData = test.collect{case x:Int => dataOnly(x)}
      //val balancedTrainingData = Balancer.balanceByPaper(trainingData, 1)
      val trainingLabels = Classifier.convertOptionalToBool(trainingData)
      Classifier.fit(trainingData.toSeq, trainingLabels)

      val currentTruthVal = Classifier.convertOptionalToBool(validationData)
      giantTruthValLabel ++= currentTruthVal

      val predValLabel = Classifier.predict(validationData)
      giantPredValLabel ++= predValLabel

      val currentTruthTest = Classifier.convertOptionalToBool(testingData)
      giantTruthTestLabel ++= currentTruthTest

      val predTestLabel = Classifier.predict(testingData)
      giantPredTestLabel ++= predTestLabel
  }

  // converts boolean mappings to 1's and 0's in validation set and then calculates metrics
  val giantTruthVal = Classifier.convertBooleansToInt(giantTruthValLabel)
  val giantPredVal = Classifier.convertBooleansToInt(giantPredValLabel)
  println(giantTruthVal.size == giantPredVal.size)
  val precisionVal = precision(giantTruthVal, giantPredVal)
  val recallVal = recall(giantTruthVal, giantPredVal)
  val f1Val = f1(giantTruthVal, giantPredVal)

  // converts boolean mappings to 1's and 0's in test set and then calculates metrics
  val giantTruthTest = Classifier.convertBooleansToInt(giantTruthTestLabel)
  val giantPredTest = Classifier.convertBooleansToInt(giantPredTestLabel)
  println(giantTruthTest.size == giantPredTest.size)
  val precisionTest = precision(giantTruthTest, giantPredTest)
  val recallTest = recall(giantTruthTest, giantPredTest)
  val f1Test = f1(giantTruthTest, giantPredTest)

  var scoreDictionary = collection.mutable.Map[String, ((String, Double, Double, Double), (String, Double, Double, Double))]()
  val dummyResults = (("validation", precisionVal, recallVal, f1Val), ("test", precisionTest, recallTest, f1Test))
  scoreDictionary += ("dummy" -> dummyResults)
  println(scoreDictionary.values)
}
