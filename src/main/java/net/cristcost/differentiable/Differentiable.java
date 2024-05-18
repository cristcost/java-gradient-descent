package net.cristcost.differentiable;

public interface Differentiable extends Chainable {
  double[] getGradient();
}
