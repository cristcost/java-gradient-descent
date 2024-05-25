package net.cristcost.jtflow.api;

public interface Chainable {

  void backpropagate(double[] outerGradient);

}
