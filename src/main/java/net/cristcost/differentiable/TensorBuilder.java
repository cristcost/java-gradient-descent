package net.cristcost.differentiable;

@FunctionalInterface
interface TensorBuilder {
  public Tensor withData(double... data);
}
