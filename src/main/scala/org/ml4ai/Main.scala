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
  //folds.foreach(x => println(x._1.mkString(" ")))
  val dataOnly = aggregatedRows.values.toSeq
  //println(folds.size)
  val giantTruthTestLabel = new mutable.HashSet[Boolean]
  val giantPredTestLabel = new mutable.HashSet[Boolean]
  val giantTruthValLabel = new mutable.HashSet[Boolean]
  val giantPredValLabel = new mutable.HashSet[Boolean]
  for((train,validate,test) <- folds) {
      val trainingData = train.map(t => dataOnly(t))
      val validationData = validate.map(v => dataOnly(v))
      val testingData = test.map(te => dataOnly(te))
      val trainingLabels = trainingData.map(x => x.label match {
        case Some(x) => x
        case _ => false
      })
      Classifier.fit(trainingData.toSeq, trainingLabels.toSeq)

      val currentTruthVal = validationData.map(v => v.label match {
        case Some(x) => x
        case _ => false
      })
      giantTruthValLabel ++= currentTruthVal

      val predValLabel = Classifier.predict(validationData)
      giantPredValLabel ++= predValLabel

      val currentTruthTest = testingData.map(t => t.label match {
        case Some(x) => x
        case _ => false
      })
      giantTruthTestLabel ++= currentTruthTest

      val predTestLabel = Classifier.predict(testingData)
      giantPredTestLabel ++= predTestLabel
  }

  // converts boolean mappings to 1's and 0's in validation set and then calculates metrics
  val giantTruthVal = Classifier.convertBooleansToInt(giantTruthValLabel.toSeq)
  val giantPredVal = Classifier.convertBooleansToInt(giantPredValLabel.toSeq)
  val precisionVal = precision(giantTruthVal, giantPredVal)
  val recallVal = recall(giantTruthVal, giantPredVal)
  val f1Val = f1(giantTruthVal, giantPredVal)

  // converts boolean mappings to 1's and 0's in test set and then calculates metrics
  val giantTruthTest = Classifier.convertBooleansToInt(giantTruthTestLabel.toSeq)
  val giantPredTest = Classifier.convertBooleansToInt(giantPredTestLabel.toSeq)
  val precisionTest = precision(giantTruthTest, giantPredTest)
  val recallTest = recall(giantTruthTest, giantPredTest)
  val f1Test = f1(giantTruthTest, giantPredTest)

  var scoreDictionary = collection.mutable.Map[String, ((String, Double, Double, Double), (String, Double, Double, Double))]()
  val dummyResults = (("validation", precisionVal, recallVal, f1Val), ("test", precisionTest, recallTest, f1Test))
  scoreDictionary += ("dummy" -> dummyResults)
  println(scoreDictionary.values)
}
