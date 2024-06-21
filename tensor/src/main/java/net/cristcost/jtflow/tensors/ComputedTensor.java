package net.cristcost.jtflow.tensors;

import java.util.Arrays;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import net.cristcost.jtflow.api.Chainable;
import net.cristcost.jtflow.api.Computation;
import net.cristcost.jtflow.api.Tensor;

@RequiredArgsConstructor
public class ComputedTensor implements Tensor, Chainable {
  @Getter
  private final double[] data;

  @Getter
  private final int[] shape;

  @Getter
  private final Computation fromComputation;

  @Override
  public void backpropagate(double[] outerGradient) {
    fromComputation.getOperation().backpropagate(outerGradient, fromComputation.getOperands());
  }

  public void startBackpropagation() {
    double[] grad = new double[size()];
    Arrays.fill(grad, 1.0);
    backpropagate(grad);
  }

  @Override
  public String toString() {
    return this.json();
  }
}
