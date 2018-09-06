package org.ml4ai.data

import java.io.InputStream

import scala.io.Source
import scala.collection.mutable.ListBuffer
case class InputRow(
                     sentenceIndex:Int,
                     PMCID:String,

                     label: Boolean,
                     EvtID: String,
                     CtxID: String,
                     closesCtxOfClass: Double,
                     context_frequency: Double,
                     evtNegationInTail: Double,
                     evtSentenceFirstPerson: Double,
                     evtSentencePastTense: Double,
                     evtSentencePresentTense: Double,
                     dependencyDistance: Double,

                     sentenceDistance: Double,
                     ctx_dependencyTails:Set[String],
                     evt_dependencyTails:Set[String]
                   )

object InputRow{
  def apply(str:String, headers: Array[String]):InputRow = {
    // Parse the commas into tokens
    //val tok = Seq.empty[String]


    val rowData = str.split(",")
    val sentencePos = rowData(0).toInt
    val listOfSpecificFeatures = List("PMCID", "label", "EvtID", "CtxID", "closesCtxOfClass", "context_frequency",
      "evtNegationInTail", "evtSentenceFirstPerson", "evtSentencePastTense", "evtSentencePresentTense", "sentenceDistance", "dependencyDistance")
    val head = headers.toSet
    val listS = headers(0) :: listOfSpecificFeatures
    val setS = listS.toSet
    val allOtherFeatures = (head -- setS).toList

    var evt_dependencyTails = new ListBuffer[String]
    var ctx_dependencyTails = new ListBuffer[String]

    for (a <- allOtherFeatures) {
      val sub = a.substring(11, a.length)

      if ((a contains "evtDepTail_") && (rowData(headers.indexOf(a)).toDouble > 0.0)) {


        evt_dependencyTails += sub
      }

      else if((a contains "ctxDepTail_") && (rowData(headers.indexOf(a)).toDouble > 0.0)) {

        ctx_dependencyTails += sub
      }


    }
    val pmcid = rowData(headers.indexOf("PMCID"))
    val label = rowData(headers.indexOf("label")).toBoolean
    val evt = rowData(headers.indexOf("EvtID"))
    val ctx = rowData(headers.indexOf("CtxID"))
    val closestCtx = rowData(headers.indexOf("closesCtxOfClass")).toDouble
    val contextFreq = rowData(headers.indexOf("context_frequency")).toDouble
    val dependencyDistance = rowData(headers.indexOf("dependencyDistance")).toDouble
    val sentenceDist = rowData(headers.indexOf("sentenceDistance")).toDouble
    val evtNegationInTail = rowData(headers.indexOf("evtNegationInTail")).toDouble
    val evtSentenceFirstPerson = rowData(headers.indexOf("evtSentenceFirstPerson")).toDouble
    val evtSentencePastTense = rowData(headers.indexOf("evtSentencePastTense")).toDouble
    val evtSentencePresentTense = rowData(headers.indexOf("evtSentencePresentTense")).toDouble
    InputRow(sentencePos, pmcid, label, evt, ctx, closestCtx, contextFreq, evtNegationInTail, evtSentenceFirstPerson, evtSentencePastTense, evtSentencePresentTense, dependencyDistance,  sentenceDist, ctx_dependencyTails.toSet, evt_dependencyTails.toSet)
  }

  def fromStream(stream:InputStream):Seq[InputRow] = {
    val source = Source.fromInputStream(stream)
    val lines = source.getLines()
    val headers = lines.next() split (",")
    val ret = (lines map (l => InputRow(l, headers))).toList
    source.close()
    ret
  }
}
