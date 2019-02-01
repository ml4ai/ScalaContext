package org.ml4ai.data.utils.correctDataPrep

import java.io.InputStream

import scala.collection.mutable
import scala.io.Source

case class AggregatedRowNew(
                             sentenceIndex:Int,
                             PMCID:String,
                             EvtID: String,
                             CtxID: String,
                             label: Option[Boolean],
                             featureGroups: Array[Double],
                             featureGroupNames:Array[String])

object AggregatedRowNew {
  private val listOfSpecificFeatures = Seq("PMCID", "label", "EvtID", "CtxID", "closesCtxOfClass_min", "closesCtxOfClass_max", "closesCtxOfClass_avg", "context_frequency_min","context_frequency_max", "context_frequency_avg",
    "evtNegationInTail_min","evtNegationInTail_max","evtNegationInTail_avg", "evtSentenceFirstPerson_min","evtSentenceFirstPerson_max","evtSentenceFirstPerson_avg", "evtSentencePastTense_min","evtSentencePastTense_max","evtSentencePastTense_avg", "evtSentencePresentTense_min","evtSentencePresentTense_max","evtSentencePresentTense_avg", "sentenceDistance_min","sentenceDistance_max","sentenceDistance_avg", "dependencyDistance_min", "dependencyDistance_max", "dependencyDistance_avg")

  private def allOtherFeatures(headers:Seq[String]): Set[String] = headers.toSet -- (listOfSpecificFeatures ++ Seq(""))
  private def indices(headers:Seq[String]): Map[String, Int] = headers.zipWithIndex.toMap
  def apply(str:String, headers: Seq[String], allOtherFeatures:Set[String], indices:Map[String, Int]):AggregatedRowNew = {
    val rowData = str.split(",")
    val sentencePos = rowData(0).toInt
    var evt_dependencyTails = new mutable.ListBuffer[Double]
    var ctx_dependencyTails = new mutable.ListBuffer[Double]
    var evt_dependencyFeatures = new mutable.ListBuffer[String]
    var ctx_dependencyFeatures = new mutable.ListBuffer[String]
    val featureGroups = new mutable.ListBuffer[Double]
    val featureNames = new mutable.ListBuffer[String]
    allOtherFeatures foreach {
      case evt:String if evt.startsWith("evtDepTail") =>
        if(rowData(indices(evt)) != "0.0")
          {evt_dependencyTails += (rowData(indices(evt))).toDouble
            evt_dependencyFeatures += evt
          }
      case ctx:String if ctx.startsWith("ctxDepTail") =>
        if(rowData(indices(ctx)) != "0.0")
        {
          ctx_dependencyTails += (rowData(indices(ctx))).toDouble
          ctx_dependencyFeatures += ctx
        }
      case _ => 0.0
    }



    val pmcid = rowData(indices("PMCID"))

    val evt = rowData(indices("EvtID"))
    val ctx = rowData(indices("CtxID"))
    val label = rowData(indices("label"))

    val listOfNumericFeatures = listOfSpecificFeatures.drop(4)
    featureNames ++= listOfNumericFeatures
    listOfNumericFeatures.map(l => {
      val tempVal = rowData(indices(l))
      featureGroups += tempVal.toDouble
    })

    featureGroups ++= evt_dependencyTails
    featureGroups ++= ctx_dependencyTails
    featureNames ++= evt_dependencyFeatures
    featureNames ++= ctx_dependencyFeatures
    AggregatedRowNew(sentencePos, pmcid, evt, ctx, Some(label.toBoolean), featureGroups.toArray, featureNames.toArray)
  }

  def fromStream(stream:InputStream):(Seq[String], Seq[AggregatedRowNew]) = {
    val source = Source.fromInputStream(stream)
    val lines = source.getLines()
    val headers = lines.next() split ","
    val features = allOtherFeatures(headers)
    val ixs = indices(headers)
    val ret = lines.map(l => AggregatedRowNew(l, headers, features, ixs)).toList
    source.close()
    (headers, ret)
  }
}
