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
  println(scoreDictionary)

  // SVM classifier
  val SVMClassifier = new LibSVMClassifier[String, String](LinearKernel)

}
