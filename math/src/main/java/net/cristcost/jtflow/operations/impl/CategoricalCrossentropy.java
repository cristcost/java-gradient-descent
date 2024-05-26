package net.cristcost.jtflow.operations.impl;

import net.cristcost.jtflow.api.Chainable;
import net.cristcost.jtflow.api.Tensor;

public class CategoricalCrossentropy {

  private static final double EPSILON = 1e-15;

  public static double[] cce(Tensor prediction, Tensor oneHotEncodedLabels) {
    validateTensorCompatibility(prediction, oneHotEncodedLabels);

    double result = cce(prediction.getData(), oneHotEncodedLabels.getData());

    return Common.makeData(result);
  }

  protected static double cce(double[] predictionData, double[] oneHotEncodedData) {
    double result = 0.0;
    for (int i = 0; i < predictionData.length; i++) {
      result -=
          oneHotEncodedData[i] * Math.log(clamp(predictionData[i], EPSILON, 1.0 - EPSILON));
    }
    return result;
  }

  private static double clamp(double inputValue, double min, double max) {
    return Math.max(min, Math.min(max, inputValue));
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

  public static void chain(double[] outerFunctionGradient, Tensor prediction,
      Tensor oneHotEncodedLabels) {

    if (prediction instanceof Chainable) {

      // CrossEntropy = − ∑ i=[0..length] (oneHotLabel[i] * log(pred[i]))
      // ∂CrossEntropy / ∂pred[i] = − (oneHotLabel[i] / pred[i])

      double[] innerGradient = new double[prediction.size()];
      for (int k = 0; k < innerGradient.length; k++) {

        double p = prediction.get(k % prediction.size());
        // As we are clamping the prediction value, the derivate varies only within this range
        if (p > EPSILON && p < 1.0 - EPSILON) {
          innerGradient[k] =
              -outerFunctionGradient[0] * oneHotEncodedLabels.get(k % oneHotEncodedLabels.size())
                  / prediction.get(k % prediction.size());
        } else {
          innerGradient[k] = 0.0;
        }
      }

      ((Chainable) prediction).backpropagate(innerGradient);
    }

    if (oneHotEncodedLabels instanceof Chainable) {
      // We could simply ignore chaining for label, but let's fail fast to allow detecting misuses
      // and eventually change the code in the future
      throw new RuntimeException("CategoricalCrossentropy Labels are not expected to be Variable.");
    }
  }
}
