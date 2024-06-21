package net.cristcost.jtflow.api;

public interface Operation {

  void backpropagate(double[] gradient, Tensor... operands);

  Tensor compute(Tensor... operands);

  String name();

}
