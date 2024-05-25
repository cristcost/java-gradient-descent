package net.cristcost.jtflow.api;

public interface Differentiable extends Chainable {
  double[] getGradient();
}
