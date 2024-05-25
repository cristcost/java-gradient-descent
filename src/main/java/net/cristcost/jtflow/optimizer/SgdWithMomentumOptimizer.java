package net.cristcost.jtflow.optimizer;

import net.cristcost.jtflow.api.Optimizer;

public class SgdWithMomentumOptimizer implements Optimizer {

  private double learningRate;
  private double momentum;
  private double[] velocity;

  public SgdWithMomentumOptimizer(double learningRate, double momentum) {
    this.learningRate = learningRate;
    this.momentum = momentum;
  }

  @Override
  public void optimize(double[] data, double[] gradient) {
    if (velocity == null) {
      velocity = new double[data.length];
    }
    for (int i = 0; i < data.length; i++) {
      velocity[i] = momentum * velocity[i] - learningRate * gradient[i];

      data[i] += velocity[i];
    }
  }

}
