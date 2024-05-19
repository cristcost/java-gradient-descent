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

  private static void validateVectorCompatibility(Tensor tensor) {
    if (tensor.getShape().length != 1) {
      // limitation of this implementation: softmax along an axis not supported
      throw new IllegalArgumentException(
          "Softmax operation requires vector of exactly 1 dimension.");
    }
  }

  public static void chain(double[] outerFunctionGradient, Tensor tensor) {

    validateVectorCompatibility(tensor);

    double[] softmax = softmax(tensor);
    
    if (tensor instanceof Chainable) {
      double[] innerGradient = new double[tensor.size()];

      for (int i = 0; i < innerGradient.length; i++) {
        for (int j = 0; j < innerGradient.length; j++) {
          double delta = (i == j) ? 1.0 : 0.0;
          innerGradient[i] += outerFunctionGradient[j] * softmax[i] * (delta - softmax[j]);
        }
      }
      ((Chainable) tensor).backpropagate(innerGradient);
    }
  }
}
