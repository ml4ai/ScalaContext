package org.ml4ai.data

object Balancer {
  /**
    *
    * @param rows Initial collection of rows
    * @param negsPerPos number of negatively labeled rows per positively labeled rows
    * @return A balanced collection of rows
    */
  def balanceByPaper(rows:Iterable[InputRow], negsPerPos:Int):Iterable[InputRow] = ???
}
