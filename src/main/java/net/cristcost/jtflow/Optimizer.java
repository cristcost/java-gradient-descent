package net.cristcost.jtflow;

public interface Optimizer {
  void optimize(double[] data, double[] gradient);
}