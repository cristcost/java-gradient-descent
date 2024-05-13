package net.cristcost.differentiable;

public interface Chainable {

  void backpropagate(double outerGradient);

}
