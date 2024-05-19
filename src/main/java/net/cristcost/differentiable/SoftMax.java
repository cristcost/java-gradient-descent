package net.cristcost.differentiable;

class SoftMax {
  static double[] softmax(Tensor a) {
    validateVectorCompatibility(a);
    double[] result = new double[a.size()];
    double sum = 0.0;

    for (int i = 0; i < a.size(); i++) {
      result[i] = Math.exp(a.get(i));
      sum += result[i];
    }
    for (int i = 0; i < a.size(); i++) {
      result[i] /= sum;
    }
    return result;
  }

  private static void validateVectorCompatibility(Tensor a) {
    if (a.getShape().length != 1) {
      // limitation of this implementation: softmax along an axis not supported
      throw new IllegalArgumentException(
          "Softmax operation requires vector of exactly 1 dimension.");
    }
  }

  public static void chain(double[] outerFunctionGradient, Tensor a) {
    validateVectorCompatibility(a);
    // if (a instanceof Chainable) {
    // double[] innerGradient = new double[a.size()];
    //
    // for (int k = 0; k < innerGradient.length; k++) {
    // innerGradient[k] = outerFunctionGradient[0] *
    // (2 * a.get(k % a.size()) - 2 * b.get(k % b.size())) / innerGradient.length;
    //
    // // ((a1 - b1)^2 + ... + (an - bn)^2) /n
    // // (a1^2 -2a1b1 + b1^2 + ... + an^2 -2anbn + bn^2) /n
    // // df/da1 (2a1 -2b1) /k
    // // df/dak (2ak -2bk) /k
    // // df/db1 (-2a1 + 2b1) /k
    // // df/dbk (-2ak + 2bk) /k
    // }
    // ((Chainable) a).backpropagate(innerGradient);
    // }
    //
    // if (b instanceof Chainable) {
    // double[] innerGradient = new double[b.size()];
    //
    // for (int k = 0; k < innerGradient.length; k++) {
    // innerGradient[k] = outerFunctionGradient[0] *
    // (2 * b.get(k % b.size()) - 2 * a.get(k % a.size())) / innerGradient.length;
    // }
    // ((Chainable) b).backpropagate(innerGradient);
    // }
  }
}
