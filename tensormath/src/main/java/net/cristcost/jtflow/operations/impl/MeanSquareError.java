package net.cristcost.jtflow.operations.impl;

import net.cristcost.jtflow.api.Tensor;
import net.cristcost.jtflow.operations.raw.RawMeanSquareError;

public class MeanSquareError {
  public static void chain(double[] outerFunctionGradient, Tensor a, Tensor b) {

    a.ifChainable(c -> c
        .backpropagate(RawMeanSquareError.gradient(outerFunctionGradient[0], a.getData(),
            b.getData())));

    b.ifChainable(c -> c.backpropagate(
        RawMeanSquareError.gradient(outerFunctionGradient[0], b.getData(),
            a.getData())));
  }

  public static double[] compute(Tensor a, Tensor b) {
    validateTensorCompatibility(a, b);
    return Common.makeData(RawMeanSquareError.compute(a.getData(), b.getData()));
  }

  public static int[] shape(Tensor tensor, Tensor other) {
    validateTensorCompatibility(tensor, other);
    return Common.SCALAR_SHAPE;
  }

  private static void validateTensorCompatibility(Tensor a, Tensor b) {
    if (a.size() != b.size()) {
      throw new IllegalArgumentException(
          "Tensor dimensions are not compatible.");
    }
  }

  // ((a1 - b1)^2 + ... + (an - bn)^2) /n
  // (a1^2 -2a1b1 + b1^2 + ... + an^2 -2anbn + bn^2) /n
  // df/da1 (2a1 -2b1) /k
  // df/dak (2ak -2bk) /k
  // df/db1 (-2a1 + 2b1) /k
  // df/dbk (-2ak + 2bk) /k


}
