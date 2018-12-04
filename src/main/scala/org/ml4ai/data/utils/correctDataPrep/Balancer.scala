package org.ml4ai.data.utils.correctDataPrep

import org.ml4ai.data.utils.oldDataPrep.InputRow

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


  def balanceByPaperAgg(aggRows:Seq[AggregatedRowNew], negsPerPos:Int): Seq[AggregatedRowNew] = {
    val groups = aggRows.groupBy(l => l.PMCID)
    val groupsById = groups.values
    var allRows:Option[Iterable[AggregatedRowNew]] = None
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
    def toBeReturned(x:Option[Iterable[AggregatedRowNew]]) = x match {
      case Some(a) => a
      case None => aggRows
    }

    toBeReturned(allRows).toSeq
  }

  private def randomRowAgg(rows: Iterable[AggregatedRowNew], negsPerPos: Int): Iterable[AggregatedRowNew] = {
    val pos_rows = rows.filter(_.label == Some(true))
    val neg_rows = rows.filter(_.label == Some(false))
    val posLength = pos_rows.size
    val negLength = neg_rows.size
    val all_rows: Iterable[AggregatedRowNew] = {
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
