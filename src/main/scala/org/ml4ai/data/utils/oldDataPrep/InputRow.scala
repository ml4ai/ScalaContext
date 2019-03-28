package org.ml4ai.data.utils.oldDataPrep

// making sure git is clean and I can commit without issues
import java.io.InputStream

import scala.collection.mutable
import scala.io.Source
case class InputRow(
                     sentenceIndex:Int,
                     PMCID:String,

                     label: Option[Boolean],
                     EvtID: String,
                     CtxID: String,
                     closesCtxOfClass: Double,
                     context_frequency: Double,
                     evtNegationInTail: Double,
                     evtSentenceFirstPerson: Double,
                     evtSentencePastTense: Double,
                     evtSentencePresentTense: Double,
                     ctxSentenceFirstPerson: Double,
                     ctxSentencePastTense: Double,
                     ctxSentencePresentTense: Double,
                     ctxNegationIntTail: Double,
                     dependencyDistance: Double,
                     sentenceDistance: Double,
                     ctx_dependencyTails:Set[String],
                     evt_dependencyTails:Set[String]
                   )

object InputRow{

  // TODO Shradha: Put this in a config file
  private val listOfSpecificFeatures = Seq("PMCID", "label", "EvtID", "CtxID", "closesCtxOfClass", "context_frequency",
    "evtNegationInTail", "evtSentenceFirstPerson", "evtSentencePastTense", "evtSentencePresentTense","ctxSentenceFirstPerson","ctxSentencePastTense", "ctxSentencePresentTense","sentenceDistance", "dependencyDistance")

  private def allOtherFeatures(headers:Seq[String]): Set[String] = headers.toSet -- (listOfSpecificFeatures ++ Seq(""))

  private def indices(headers:Seq[String]): Map[String, Int] = headers.zipWithIndex.toMap

  def apply(str:String, headers: Seq[String], allOtherFeatures:Set[String], indices:Map[String, Int]):InputRow = {
    // Parse the commas into tokens
    val rowData = str.split(",")
    val sentencePos = rowData(0).toInt


    var evt_dependencyTails = new mutable.HashSet[String]
    var ctx_dependencyTails = new mutable.HashSet[String]
    
    allOtherFeatures foreach {
      case evt:String if evt.startsWith("evtDepTail") =>
        if(rowData(indices(evt)) != "0.0")
          evt_dependencyTails += evt.substring(11)
      case ctx:String if ctx.startsWith("ctxDepTail") =>
        if(rowData(indices(ctx)) != "0.0")
          ctx_dependencyTails += ctx.substring(11)
      case _ => ()
    }

    val pmcid = rowData(indices("PMCID"))
    val label = rowData(indices("label"))
    val evt = rowData(indices("EvtID"))
    val ctx = rowData(indices("CtxID"))
    val closestCtx = rowData(indices("closesCtxOfClass"))
    val contextFreq = rowData(indices("context_frequency"))
    val dependencyDistance = rowData(indices("dependencyDistance"))
    val sentenceDist = rowData(indices("sentenceDistance"))
    val evtNegationInTail = rowData(indices("evtNegationInTail"))
    val evtSentenceFirstPerson = rowData(indices("evtSentenceFirstPerson"))
    val evtSentencePastTense = rowData(indices("evtSentencePastTense"))
    val evtSentencePresentTense = rowData(indices("evtSentencePresentTense"))
    val ctxSentenceFirstPerson = rowData(indices("ctxSentenceFirstPerson"))
    val ctxSentencePastTense = rowData(indices("ctxSentencePastTense"))
    val ctxSentencePresentTense = rowData(indices("ctxSentencePresentTense"))
    val ctxNegationIntTail = rowData(indices("ctxNegationIntTail"))
    InputRow(sentencePos,
      pmcid,
      Some(label.toBoolean),
      evt,
      ctx,
      closestCtx.toDouble,
      contextFreq.toDouble,
      evtNegationInTail.toDouble,
      evtSentenceFirstPerson.toDouble,
      evtSentencePastTense.toDouble,
      evtSentencePresentTense.toDouble,
      ctxSentenceFirstPerson.toDouble,
      ctxSentencePastTense.toDouble,
      ctxSentencePresentTense.toDouble,
      ctxNegationIntTail.toDouble,
      dependencyDistance.toDouble,
      sentenceDist.toDouble,
      ctx_dependencyTails.toSet,
      evt_dependencyTails.toSet)
  }

  def fromStream(stream:InputStream):Seq[InputRow] = {
    val source = Source.fromInputStream(stream)
    val lines = source.getLines()
    val headers = lines.next() split ","
    val features = allOtherFeatures(headers)
    val ixs = indices(headers)
    val ret = (lines map (l => InputRow(l, headers.toSeq, features, ixs))).toList
    source.close()
    ret
  }

}
