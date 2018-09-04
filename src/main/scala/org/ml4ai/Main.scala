package org.ml4ai

import java.util.zip.GZIPInputStream

import scala.io.Source
import data.InputRow


object Main extends App {

  val rows = InputRow.fromStream(new GZIPInputStream(getClass.getResourceAsStream("/features.csv.gz")))

  rows foreach println

}
