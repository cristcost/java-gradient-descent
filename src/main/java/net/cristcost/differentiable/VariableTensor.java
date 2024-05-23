package net.cristcost.differentiable;

import lombok.Getter;

public class VariableTensor implements Tensor, Differentiable {

  @Getter
  private final double[] data;

  @Getter
  private final int[] shape;

  @Getter
  private final double[] gradient;
  
  void set(double value, int... indices) {
    int index = Tensor.calculateIndex(shape, indices);
    if (index >= size()) {
      throw new ArrayIndexOutOfBoundsException(
          String.format(
              "Requested index is beyond the size of the tensor data: result index %d >= size %d",
              index, size()));
    }
    data[index % data.length] = value;
  }
  
  @Override
  public String toString() {
    return this.json();
  }

  public VariableTensor(double[] data, int[] shape) {
    this.data = data;
    this.shape = shape;
    this.gradient = new double[data.length];
  }

  @Override
  public void backpropagate(double[] outerGradient) {
    for (int i = 0; i < outerGradient.length; i++) {
      this.gradient[i] += outerGradient[i];
    }
  }

}
