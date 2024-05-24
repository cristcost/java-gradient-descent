package net.cristcost.differentiable;

public class RootMeanSquarePropagationOptimizer implements Optimizer {


  private double learningRate;
  private double decayRate;
  private double[] squaredGradientCache;

  public RootMeanSquarePropagationOptimizer(double learningRate, double decayRate) {
    this.learningRate = learningRate;
    this.decayRate = decayRate;
  }

  @Override
  public void optimize(double[] data, double[] gradient) {
    if (squaredGradientCache == null) {
      squaredGradientCache = new double[data.length];
    }
    for (int i = 0; i < data.length; i++) {
      squaredGradientCache[i] =
          decayRate * squaredGradientCache[i] + (1 - decayRate) * gradient[i] * gradient[i];

      data[i] -=
          learningRate * gradient[i] / (Math.sqrt(squaredGradientCache[i]) + 1e-8);
    }
  }
}
