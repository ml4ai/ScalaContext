package org.ml4ai

import java.util.zip.GZIPInputStream


import data.InputRow
import data.Balancer
import data.AggregatedRow
import data.FoldMaker
import data.DummyClassifier
import scala.collection.mutable
import smile.validation._
import data.GBRT
object Main extends App {

  val rows = InputRow.fromStream(new GZIPInputStream(getClass.getResourceAsStream("/features.csv.gz")))
  val balancedRows = Balancer.balanceByPaper(rows, 1)
  val possibleFeatures = FoldMaker.getPossibleNumericFeatures(balancedRows)
  val aggregatedRows = AggregatedRow.fromRows(balancedRows)
  val folds = FoldMaker.getFolds(aggregatedRows)
  val retainKeys = aggregatedRows mapValues(x => x)
  val mapEntrySeq = retainKeys.toSeq
  val dataOnly = mapEntrySeq.map(x => x._2)
  val giantTruthTestLabel = new mutable.ArrayBuffer[Int]()
  val giantPredTestLabel = new mutable.ArrayBuffer[Int]()
  val giantTruthValLabel = new mutable.ArrayBuffer[Int]()
  val giantPredValLabel = new mutable.ArrayBuffer[Int]()
  for((train,validate,test) <- folds) {
      val trainingData = train.collect{case x:Int => mapEntrySeq(x)}
      val balancedTrainingData = Balancer.balanceByPaperAgg(trainingData, 1)
      val balancedTrainingFrame = FoldMaker.createData(possibleFeatures, balancedTrainingData)
      val trainingLabels = DummyClassifier.convertOptionalToBool(balancedTrainingData.toSeq)
      val labelsToInt = DummyClassifier.convertBooleansToInt(trainingLabels)

      val validationData = validate.collect{case x:Int => dataOnly(x)}
      val validationFrame = FoldMaker.createData(possibleFeatures, validationData)

      val testingData = test.collect{case x:Int => dataOnly(x)}
      val testingFrame = FoldMaker.createData(possibleFeatures, testingData)


      DummyClassifier.fit(balancedTrainingFrame, labelsToInt)

      val currentTruthVal = DummyClassifier.convertOptionalToBool(validationData)
      val currentTruthInt = DummyClassifier.convertBooleansToInt(currentTruthVal)
      giantTruthValLabel ++= currentTruthInt

      val predValLabel = DummyClassifier.predict(validationFrame)

      giantPredValLabel ++= predValLabel
      val currentTruthTest = DummyClassifier.convertOptionalToBool(testingData)
      val currentTruthTestInt = DummyClassifier.convertBooleansToInt(currentTruthTest)
      giantTruthTestLabel ++= currentTruthTestInt

      val predTestLabel = DummyClassifier.predict(testingFrame)

      giantPredTestLabel ++= predTestLabel
  }

  // converts boolean mappings to 1's and 0's in validation set and then calculates metrics
  println(giantTruthValLabel.size == giantPredValLabel.size)
  val precisionVal = precision(giantTruthValLabel.toArray, giantPredValLabel.toArray)
  val recallVal = recall(giantTruthValLabel.toArray, giantPredValLabel.toArray)
  val f1Val = f1(giantTruthValLabel.toArray, giantPredValLabel.toArray)

  // converts boolean mappings to 1's and 0's in test set and then calculates metrics
  println(giantTruthTestLabel.size == giantPredTestLabel.size)
  val precisionTest = precision(giantTruthTestLabel.toArray, giantPredTestLabel.toArray)
  val recallTest = recall(giantTruthTestLabel.toArray, giantPredTestLabel.toArray)
  val f1Test = f1(giantTruthTestLabel.toArray, giantPredTestLabel.toArray)

  var scoreDictionary = collection.mutable.Map[String, ((String, Double, Double, Double), (String, Double, Double, Double))]()
  val dummyResults = (("validation", precisionVal, recallVal, f1Val), ("test", precisionTest, recallTest, f1Test))
  scoreDictionary += ("dummy" -> dummyResults)
  println(scoreDictionary.values)



}
