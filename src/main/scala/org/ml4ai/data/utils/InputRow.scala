package org.ml4ai.data.utils
import com.typesafe.config.ConfigFactory
import java.io.InputStream

import scala.collection.mutable
import scala.io.Source
case class InputRow(
                     sentenceIndex:Int,
                     PMCID:String,

                     label: Option[Boolean],
                     EvtID: String,
                     CtxID: String,
                     specificFeatureNames:Array[String],
                     specificFeatureValues:Array[Double],
                     ctx_dependencyTails:Set[String],
                     evt_dependencyTails:Set[String]
                   )

object InputRow{

  val config = ConfigFactory.load()
  val hardCodedInputRowFeatures = config.getString("features.hardCodedInputRowFeatures")
  private val listOfSpecificFeatures = CodeUtils.readHardcodedFeaturesFromFile(hardCodedInputRowFeatures)
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

    val specificFeatureNames = collection.mutable.ListBuffer[String]()
    val specificFeatureValues = collection.mutable.ListBuffer[Double]()
    val listOfNumericFeatures = listOfSpecificFeatures.drop(4)
    listOfNumericFeatures.map(l => {
      specificFeatureNames += l
      val value = rowData(indices(l))
      specificFeatureValues += value.toDouble
    })
    InputRow(sentencePos,
      pmcid,
      Some(label.toBoolean),
      evt,
      ctx,
      specificFeatureNames.toArray,
      specificFeatureValues.toArray,
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
