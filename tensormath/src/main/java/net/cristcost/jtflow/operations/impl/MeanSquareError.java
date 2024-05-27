package net.cristcost.jtflow.operations.impl;

import net.cristcost.jtflow.api.Chainable;
import net.cristcost.jtflow.api.Tensor;

public class MeanSquareError {
  public static double[] mse(Tensor a, Tensor b) {
    validateTensorCompatibility(a, b);
    return Common.makeData(mse(a.getData(), b.getData()));
  }

  protected static double mse(double[] aData, double[] bData) {
    double result = 0.0;

    for (int i = 0; i < aData.length; i++) {
      result += Math.pow(aData[i % aData.length] - bData[i % bData.length], 2.0);
    }
    result /= aData.length;
    return result;
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

  public static void chain(double[] outerFunctionGradient, Tensor a, Tensor b) {

    if (a instanceof Chainable) {
      ((Chainable) a)
          .backpropagate(operandGradient(outerFunctionGradient[0], a.getData(), b.getData()));
    }

    if (b instanceof Chainable) {
      ((Chainable) b)
          .backpropagate(
              operandGradient(outerFunctionGradient[0], b.getData(), a.getData()));
    }
  }

  // ((a1 - b1)^2 + ... + (an - bn)^2) /n
  // (a1^2 -2a1b1 + b1^2 + ... + an^2 -2anbn + bn^2) /n
  // df/da1 (2a1 -2b1) /k
  // df/dak (2ak -2bk) /k
  // df/db1 (-2a1 + 2b1) /k
  // df/dbk (-2ak + 2bk) /k


  protected static double[] operandGradient(double outerFunctionGradient,
      double[] variantData,
      double[] invariantData) {
    double[] innerGradient = new double[variantData.length];

    for (int k = 0; k < innerGradient.length; k++) {
      innerGradient[k] = outerFunctionGradient *
          (2 * variantData[k % variantData.length] - 2 * invariantData[k % invariantData.length])
          / innerGradient.length;

    }
    return innerGradient;
  }
}
