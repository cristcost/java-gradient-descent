package net.cristcost.jtflow.operations.impl;

import net.cristcost.jtflow.JTFlow;
import net.cristcost.jtflow.api.Chainable;
import net.cristcost.jtflow.api.Tensor;

public class MeanSquareError {
  public static double[] mse(Tensor a, Tensor b) {
    validateTensorCompatibility(a, b);

    double result = 0.0;

    for (int i = 0; i < a.size(); i++) {
      result += Math.pow(a.get(i) - b.get(i), 2.0);
    }
    result /= a.size();

    return JTFlow.data(result);
  }

  public static int[] shape(Tensor tensor, Tensor other) {
    validateTensorCompatibility(tensor, other);
    return JTFlow.shape();
  }

  private static void validateTensorCompatibility(Tensor a, Tensor b) {
    if (a.size() != b.size()) {
      throw new IllegalArgumentException(
          "Tensor dimensions are not compatible.");
    }
  }

  public static void chain(double[] outerFunctionGradient, Tensor a, Tensor b) {

    if (a instanceof Chainable) {
      double[] innerGradient = new double[a.size()];

      for (int k = 0; k < innerGradient.length; k++) {
        innerGradient[k] = outerFunctionGradient[0] *
            (2 * a.get(k % a.size()) - 2 * b.get(k % b.size())) / innerGradient.length;

        // ((a1 - b1)^2 + ... + (an - bn)^2) /n
        // (a1^2 -2a1b1 + b1^2 + ... + an^2 -2anbn + bn^2) /n
        // df/da1 (2a1 -2b1) /k
        // df/dak (2ak -2bk) /k
        // df/db1 (-2a1 + 2b1) /k
        // df/dbk (-2ak + 2bk) /k
      }
      ((Chainable) a).backpropagate(innerGradient);
    }

    if (b instanceof Chainable) {
      double[] innerGradient = new double[b.size()];

      for (int k = 0; k < innerGradient.length; k++) {
        innerGradient[k] = outerFunctionGradient[0] *
            (2 * b.get(k % b.size()) - 2 * a.get(k % a.size())) / innerGradient.length;
      }
      ((Chainable) b).backpropagate(innerGradient);
    }
  }
}
