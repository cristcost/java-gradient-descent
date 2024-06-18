package net.cristcost.jtflow.tensors;

import java.util.Arrays;
import lombok.Getter;
import lombok.Setter;
import net.cristcost.jtflow.api.Differentiable;
import net.cristcost.jtflow.api.Optimizer;
import net.cristcost.jtflow.api.Tensor;

public class VariableTensor implements Tensor, Differentiable {

  @Getter
  private final double[] data;

  @Getter
  private final int[] shape;

  @Getter
  private final double[] gradient;

  @Setter
  private Optimizer optimizer;

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

  public void optimize() {
    if (optimizer == null) {
      throw new RuntimeException("Missing optimizer: Cannot optimize tensor");
    }
    optimizer.optimize(data, gradient);
    Arrays.fill(gradient, 0.0);
  }

  @Override
  public String toString() {
    return this.json();
  }

  void set(double value, int... indices) {
    int index = calculateIndex(indices);
    if (index >= size()) {
      throw new ArrayIndexOutOfBoundsException(
          String.format(
              "Requested index is beyond the size of the tensor data: result index %d >= size %d",
              index, size()));
    }
    data[index % data.length] = value;
  }

}
