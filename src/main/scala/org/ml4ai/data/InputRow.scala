package org.ml4ai.data

import java.io.InputStream

import scala.io.Source

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
                     ctx_dependencyTails:List[String],
                     evt_dependencyTails:List[String]
                   )

object InputRow{
  def apply(str:String, headers: Array[String]):InputRow = {
    // Parse the commas into tokens
    //val tok = Seq.empty[String]
    //println(str)

    val rowData = str.split(",")
    val sentencePos = rowData(0).toInt
    val listOfSpecificFeatures = List("PMCID", "label", "EvtID", "CtxID", "closesCtxOfClass", "context_frequency",
      "evtNegationInTail", "evtSentenceFirstPerson", "evtSentencePastTense", "evtSentencePresentTense", "sentenceDistance", "dependencyDistance")
    val head = headers.toSet
    val listS = headers(0) :: listOfSpecificFeatures
    val setS = listS.toSet
    val allOtherFeatures = (head -- setS).toList
    var evt_dependencyTails = List()
    var ctx_depenedencyTails = List()

    for (a <- allOtherFeatures) {
      if ((a contains "evtDepTail_") && (rowData(headers.indexOf(a)).toDouble > 0.0)) {
        val sub = a.substring(11, a.length -1)
        sub :: evt_dependencyTails
      }

      else if((a contains "ctxDepTail_") && (rowData(headers.indexOf(a)).toDouble > 0.0)) {
        //println(a)
      }
        //a :: ctx_depenedencyTails

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
    InputRow(sentencePos, pmcid, label, evt, ctx, closestCtx, contextFreq, evtNegationInTail, evtSentenceFirstPerson, evtSentencePastTense, evtSentencePresentTense, dependencyDistance,  sentenceDist, List(), List())

    //InputRow(sentencePos, pmcid, label, evt, ctx, closestCtx, contextFreq, evtNegationInTail, evtSentenceFirstPerson, evtSentencePastTense, evtSentencePresentTense, dependencyDistance,  sentenceDist, ctx_depenedencyTails, evt_dependencyTails)
  }

  def fromStream(stream:InputStream):Seq[InputRow] = {
    val source = Source.fromInputStream(stream)
    val lines = source.getLines()
    val headers = lines.next() split (",")
    //headers foreach println
    val ret = (lines map (l => InputRow(l, headers))).toList
    source.close()
    ret
  }
}