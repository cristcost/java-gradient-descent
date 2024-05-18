package net.cristcost.differentiable;

class DotProduct {
  static double[] dot(Tensor a, Tensor b) {
    validateVectorCompatibility(a, b);

    double result = 0.0;

    for (int i = 0; i < a.size(); i++) {
      result += a.get(i) * b.get(i);
    }

    return MathLibrary.data(result);
  }

  static int[] shape(Tensor tensor, Tensor other) {
    validateVectorCompatibility(tensor, other);
    return MathLibrary.shape();
  }

  private static void validateVectorCompatibility(Tensor a, Tensor b) {
    // Check if both arrays have at least 2 dimensions
    if (a.getShape().length != 1 || b.getShape().length != 1) {
      throw new IllegalArgumentException(
          "Dot product operation requires vector of exavtly 1 dimension.");
    }

    if (a.getShape()[0] != b.getShape()[0]) {
      throw new IllegalArgumentException(
          "Vector dimensions are not compatible for dot product.");
    }
  }

  public static void chain(double[] outerFunctionGradient, Tensor a, Tensor b) {

    if (a instanceof Chainable) {
      double[] innerGradient = new double[a.size()];

      for (int k = 0; k < innerGradient.length; k++) {
        innerGradient[k] = outerFunctionGradient[0] * b.get(k % b.size());
      }
      ((Chainable) a).backpropagate(innerGradient);
    }

    if (b instanceof Chainable) {
      double[] innerGradient = new double[b.size()];

      for (int k = 0; k < innerGradient.length; k++) {
        innerGradient[k] = outerFunctionGradient[0] * a.get(k % a.size());
      }
      ((Chainable) b).backpropagate(innerGradient);
    }
  }
}
