package org.ml4ai

import java.util.zip._


import data.InputRow
import data.Balancer
import data.AggregatedRow
import data.FoldMaker
import data.DummyClassifier
import scala.collection.mutable
import data.Utils
import data.Baseline
import scala.io.Source
object Main extends App {

  val rows = InputRow.fromStream(new GZIPInputStream(getClass.getResourceAsStream("/features.csv.gz")))
  val balancedRows = Balancer.balanceByPaper(rows, 1)
  val possibleFeatures = FoldMaker.getPossibleNumericFeatures(balancedRows)
  println("number of rows before aggregation : " + balancedRows.size)
  val aggregatedRows = AggregatedRow.fromRows(balancedRows)
  val bufferedFoldIndices = Source.fromFile("./src/main/resources/cv_folds_val_4.csv")
  val foldsFromCSV = FoldMaker.getFoldsPerPaper(bufferedFoldIndices)
  println("number of rows after aggregation : " + aggregatedRows.size)
  val retainKeys = aggregatedRows mapValues(x => x)
  val mapEntrySeq = retainKeys.toSeq
  val dataOnly = mapEntrySeq.map(x => x._2)
  val giantTruthTestLabel = new mutable.ArrayBuffer[Int]()
  val giantPredTestLabel = new mutable.ArrayBuffer[Int]()
  val giantTruthValLabel = new mutable.ArrayBuffer[Int]()
  val giantPredValLabel = new mutable.ArrayBuffer[Int]()
  for((train,_,test) <- foldsFromCSV) {
      val trainingData = train.collect{case x:Int => mapEntrySeq(x)}
      val sentDistFrame = FoldMaker.createSentenceDistData(trainingData)
      val balancedTrainingData = Balancer.balanceByPaperAgg(sentDistFrame, 1)
      val balancedTrainingFrame = FoldMaker.createData(Seq("sentenceDistance_min"), balancedTrainingData)
      val trainingLabels = DummyClassifier.convertOptionalToBool(balancedTrainingData.toSeq)
      val labelsToInt = DummyClassifier.convertBooleansToInt(trainingLabels)
      val kToF1Map = new mutable.HashMap[Int, Double]
      for(k_val <- 0 until 51) {
        val trainInstance = new Baseline(k_val)
        val pred = trainInstance.predict(balancedTrainingFrame)
        val counts = Utils.predictCounts(labelsToInt, pred)
        val f1score = Utils.f1(counts)
        kToF1Map += (k_val -> f1score)
      }

      val bestK = Utils.argMax(kToF1Map.toMap)
      val testingData = test.collect{case x:Int => mapEntrySeq(x)}
      val testSentFrame = FoldMaker.createSentenceDistData(testingData)
      val extractCont = testSentFrame.map(x => x._2)
      val testingFrame = FoldMaker.createData(Seq("sentenceDistance_min"), extractCont)
      val currentTruthTest = DummyClassifier.convertOptionalToBool(extractCont.toSeq)
      val currentTruthTestInt = DummyClassifier.convertBooleansToInt(currentTruthTest)
      giantTruthTestLabel ++= currentTruthTestInt
      val testInstance = new Baseline(bestK)
      val predTestLabel = testInstance.predict(testingFrame)
      giantPredTestLabel ++= predTestLabel

  }

  // converts boolean mappings to 1's and 0's in test set and then calculates metrics

  val testCounts = Utils.predictCounts(giantTruthTestLabel.toArray, giantPredTestLabel.toArray)
  val precisionTest = Utils.precision(testCounts)
  val recallTest = Utils.recall(testCounts)
  val f1Test = Utils.f1(testCounts)


  var scoreDictionary = collection.mutable.Map[String, ((String, Double, Double, Double), (String, Double, Double, Double))]()
  val baselineResults = (("validation", 0.0,0.0,0.0), ("test", precisionTest, recallTest, f1Test))
  scoreDictionary += ("baseline" -> baselineResults)
  println(scoreDictionary.values)

}
