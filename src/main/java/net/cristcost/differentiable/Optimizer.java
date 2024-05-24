package net.cristcost.differentiable;

public interface Optimizer {
  void optimize(double[] data, double[] gradient);
}