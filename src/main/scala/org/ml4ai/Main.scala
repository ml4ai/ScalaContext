package org.ml4ai

import java.util.zip.GZIPInputStream


import data.InputRow
import data.Balancer

object Main extends App {

  val rows = InputRow.fromStream(new GZIPInputStream(getClass.getResourceAsStream("/features.csv.gz")))
  val balancedRows = Balancer.balanceByPaper(rows, 1)
  balancedRows foreach println

}
