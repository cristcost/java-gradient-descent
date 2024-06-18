package net.cristcost.jtflow.api;

public interface Optimizer {
  void optimize(double[] parameters, double[] parametersGradient);
}
