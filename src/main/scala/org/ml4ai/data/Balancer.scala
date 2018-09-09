package org.ml4ai.data

object Balancer {
  /**
    *
    * @param rows       Initial collection of rows
    * @param negsPerPos number of negatively labeled rows per positively labeled rows
    * @return A balanced collection of rows
    */
  def balanceByPaper(rows: Iterable[InputRow], negsPerPos: Int): Iterable[InputRow] = {
    val randomRows = randomRowSelection(rows, negsPerPos)
    //randomRows foreach println
    rows

  }

  def randomRowSelection(rows: Iterable[InputRow], negsPerPos: Int): Iterable[InputRow] = {
    val pos_rows = rows.filter(_.label == Some(true))
    val neg_rows = rows.filter(_.label == Some(false))
    val posLength = pos_rows.size
    val negLength = neg_rows.size
    val all_rows: Iterable[InputRow] = {
      if (negLength < posLength) {
        val numOfPos = negLength * negsPerPos
        if(numOfPos > posLength)
          throw new IllegalArgumentException("Requested balancing requires more pos examples than total present.")
        val shuffled = scala.util.Random.shuffle(pos_rows.toList)
        val subShuffled = shuffled.filter(shuffled.indexOf(_) <= numOfPos - 1)
        neg_rows.toList ::: subShuffled
      }
      else {
        val numOfNeg = posLength * negsPerPos
        if(numOfNeg > negLength)
          throw new IllegalArgumentException("Requested balancing requires more neg examples than total present.")
        val shuffled = scala.util.Random.shuffle(neg_rows.toList)
        val subShuffled = shuffled.filter(shuffled.indexOf(_) <= numOfNeg - 1)
        pos_rows.toList ::: subShuffled
      }
    }
    all_rows
  }
}

