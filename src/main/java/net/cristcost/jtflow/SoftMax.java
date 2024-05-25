package net.cristcost.jtflow;

import java.util.Arrays;

class SoftMax {
  static double[] softmax(Tensor a) {
    validateVectorCompatibility(a);
    double[] result = new double[a.size()];
    double sum = 0.0;

    // this is avoid Numerical instability large exponent numbers, we shift the values used for
    // computation by the max. We can do that as Softmax function is invariant to translation by a
    // constant
    double numericalInstabilityCorrection =
        Arrays.stream(a.getData()).max().orElse(Double.NEGATIVE_INFINITY);

    for (int i = 0; i < a.size(); i++) {
      result[i] = Math.exp(a.get(i) - numericalInstabilityCorrection);
      sum += result[i];
    }
    for (int i = 0; i < a.size(); i++) {
      result[i] /= sum;
    }

    return result;
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

  public static void chain(double[] outerFunctionGradient, Tensor tensor) {

    validateVectorCompatibility(tensor);

    if (tensor instanceof Chainable) {
      double[] softmax = new double[tensor.size()];
      double[] innerGradient = new double[tensor.size()];
      double sum = 0.0;

      // this is avoid Numerical instability large exponent numbers, we shift the values used for
      // computation by the max. We can do that as Softmax function is invariant to translation by a
      // constant AND chain rule is based on the softmax
      double numericalInstabilityCorrection =
          Arrays.stream(tensor.getData()).max().orElse(Double.NEGATIVE_INFINITY);

      for (int i = 0; i < tensor.size(); i++) {
        softmax[i] = Math.exp(tensor.get(i) - numericalInstabilityCorrection);
        sum += softmax[i];
      }

      for (int i = 0; i < innerGradient.length; i++) {
        for (int j = 0; j < innerGradient.length; j++) {
          double delta = (i == j) ? 1.0 : 0.0;
          innerGradient[i] +=
              outerFunctionGradient[j] * softmax[i] / sum * (delta - softmax[j] / sum);
        }
      }
      ((Chainable) tensor).backpropagate(innerGradient);
    }
  }
}
