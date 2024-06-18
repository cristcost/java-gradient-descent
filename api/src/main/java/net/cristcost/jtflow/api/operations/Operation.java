package net.cristcost.jtflow.api.operations;

import net.cristcost.jtflow.api.Tensor;

public interface Operation {

  Tensor compute(Tensor... operands);

  void backpropagate(double[] gradient, Tensor... operands);
  
  String name();
 
}
