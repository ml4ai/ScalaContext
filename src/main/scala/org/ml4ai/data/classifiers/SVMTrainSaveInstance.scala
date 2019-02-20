package org.ml4ai.data.classifiers

import java.util.zip.GZIPInputStream

import org.clulab.learning.LinearSVMClassifier
import org.ml4ai.data.utils.correctDataPrep.AggregatedRowNew

object SVMTrainSaveInstance extends App {
  val fileName = "./src/main/resources/svmTrainedModel.dat"
  val SVMClassifier = new LinearSVMClassifier[Int, String](C = 0.001, eps = 0.001, bias = false)
  val svmInstance = new LinearSVMWrapper(SVMClassifier)
  val (allFeatures,rows) = AggregatedRowNew.fromStream(new GZIPInputStream(getClass.getResourceAsStream("/grouped_features.csv.gz")))
  val trainingData = rows.filter(_.PMCID != "b'PMC4204162'")
  val trainingLabels = DummyClassifier.convertOptionalToBool(trainingData)
  val labelsToInt = DummyClassifier.convertBooleansToInt(trainingLabels)
  val tups = svmInstance.constructTupsForRVF(trainingData)
  val (trainDataSet, _) = svmInstance.mkRVFDataSet(labelsToInt,tups)
  svmInstance.fit(trainDataSet)
  svmInstance.saveModel(fileName)
}
