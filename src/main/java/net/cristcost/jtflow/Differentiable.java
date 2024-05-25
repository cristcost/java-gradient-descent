package net.cristcost.jtflow;

public interface Differentiable extends Chainable {
  double[] getGradient();
}
