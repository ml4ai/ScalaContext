package org.ml4ai.data

object Balancer {
  /**
    *
    * @param rows       Initial collection of rows
    * @param negsPerPos number of negatively labeled rows per positively labeled rows
    * @return A balanced collection of rows
    */
  def balanceByPaper(rows: Iterable[InputRow], negsPerPos: Int): Iterable[InputRow] = {

    val groups = rows.groupBy(_.PMCID)
    val groupByPMCID = groups.values
    var allRows:Option[Iterable[InputRow]] = None
    for (g <- groupByPMCID) {
      allRows match {
        case None => {
          val random = randomRowSelection(g, negsPerPos)
          allRows = Some(random)
        }
        case Some(s) => {
          val chosenRows = randomRowSelection(g, negsPerPos)
          allRows = Some(s ++ chosenRows)
        }
      }
    }

    def toBeReturned(x:Option[Iterable[InputRow]]) = x match {
      case Some(a) => a
      case None => rows
    }

    toBeReturned(allRows)

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
        val subShuffled = shuffled.slice(0,numOfPos)
        neg_rows.toList ::: subShuffled
      }
      else {
        val numOfNeg = posLength * negsPerPos
        if(numOfNeg > negLength)
          throw new IllegalArgumentException("Requested balancing requires more neg examples than total present.")
        val shuffled = scala.util.Random.shuffle(neg_rows.toList)
        val subShuffled = shuffled.slice(0,numOfNeg)
        pos_rows.toList ::: subShuffled
      }
    }
    all_rows

  }


  def balanceByPaperAgg(aggRows:Iterable[((String, String, String), AggregatedRow)], negsPerPos:Int): Iterable[AggregatedRow] = {
    val groups = aggRows.groupBy(l => l._1._1)
    val groupsById = groups.values
    var allRows:Option[Iterable[AggregatedRow]] = None
    for(g <- groupsById) {
      allRows match {
        case None => val random = randomRowAgg(g, negsPerPos)
          allRows = Some(random)
        case Some(s) => {
          val chosenRows = randomRowAgg(g, negsPerPos)
          allRows = Some(s ++ chosenRows)
        }
      }
    }
    def toBeReturned(x:Option[Iterable[AggregatedRow]]) = x match {
      case Some(a) => a
      case None => aggRows.map(x => x._2)
    }

    toBeReturned(allRows)
  }

  private def randomRowAgg(rows: Iterable[((String, String, String), AggregatedRow)], negsPerPos: Int): Iterable[AggregatedRow] = {
    val pos_rows = rows.filter(_._2.label == Some(true)).map(x => x._2)
    val neg_rows = rows.filter(_._2.label == Some(false)).map(x => x._2)
    val posLength = pos_rows.size
    val negLength = neg_rows.size
    val all_rows: Iterable[AggregatedRow] = {
      if (negLength < posLength) {
        val numOfPos = negLength * negsPerPos
        if(numOfPos > posLength)
          throw new IllegalArgumentException("Requested balancing requires more pos examples than total present.")
        val shuffled = scala.util.Random.shuffle(pos_rows.toList)
        val subShuffled = shuffled.slice(0,numOfPos)
        neg_rows.toList ::: subShuffled
      }
      else {
        val numOfNeg = posLength * negsPerPos
        if(numOfNeg > negLength)
          throw new IllegalArgumentException("Requested balancing requires more neg examples than total present.")
        val shuffled = scala.util.Random.shuffle(neg_rows.toList)
        val subShuffled = shuffled.slice(0,numOfNeg)
        pos_rows.toList ::: subShuffled
      }

    }
    all_rows
  }
}

