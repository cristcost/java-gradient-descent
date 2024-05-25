package net.cristcost.jtflow;

public interface Chainable {

  void backpropagate(double[] outerGradient);

}
