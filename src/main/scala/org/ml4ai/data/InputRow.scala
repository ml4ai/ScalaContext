package org.ml4ai.data

import java.io.InputStream

import scala.io.Source

case class InputRow(
                   pmcid:String,
                   sentence:Int,
                   dependencyTails:Set[String]
                   )

object InputRow{
  def apply(str:String):InputRow = {
    // Parse the commas into tokens
    //val tok = Seq.empty[String]
    InputRow("hello", 1, Set())
  }

  def fromStream(stream:InputStream):Seq[InputRow] = {
    val source = Source.fromInputStream(stream)
    val lines = source.getLines()
    val headers = lines.next() split (",")
    val ret = (lines map (l => InputRow(l))).toList
    source.close()
    ret
  }
}
