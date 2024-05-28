package net.cristcost.jtflow.operations.impl;

import net.cristcost.jtflow.api.Chainable;
import net.cristcost.jtflow.api.Tensor;
import net.cristcost.jtflow.operations.raw.RawSoftMax;


public class SoftMax {
  public static void chain(double[] outerFunctionGradient, Tensor tensor) {

    validateVectorCompatibility(tensor);

    if (tensor instanceof Chainable) {
      ((Chainable) tensor)
          .backpropagate(RawSoftMax.gradient(outerFunctionGradient, tensor.getData()));
    }
  }

  public static double[] compute(Tensor a) {
    validateVectorCompatibility(a);
    return RawSoftMax.compute(a.getData());
  }

  private static void validateVectorCompatibility(Tensor tensor) {
    int non1Dims = 0;
    for (int dim : tensor.getShape()) {
      if (dim != 1) {
        non1Dims++;
      }
    }

    if (non1Dims > 1) {
      // limitation of this implementation: softmax along an axis not supported
      throw new IllegalArgumentException(
          "Softmax operation requires vector of exactly 1 dimension.");
    }
  }
}
