package net.cristcost.jtflow.api.operations;

import net.cristcost.jtflow.api.Tensor;

public interface Operation {

  void backpropagate(double[] gradient, Tensor... operands);

  Tensor compute(Tensor... operands);

  String name();

}
