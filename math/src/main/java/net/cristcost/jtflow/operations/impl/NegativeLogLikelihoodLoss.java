package net.cristcost.jtflow.operations.impl;

import net.cristcost.jtflow.api.Chainable;
import net.cristcost.jtflow.api.Tensor;

public class NegativeLogLikelihoodLoss {

  private static final double EPSILON = 1e-15;

  public static double[] nll(Tensor prediction, Tensor oneHotEncodedLabels) {
    validateTensorCompatibility(prediction, oneHotEncodedLabels);

    double result = nll(prediction.getData(), oneHotEncodedLabels.getData());

    return Common.makeData(result);
  }

  protected static double nll(double[] predictionData, double[] oneHotEncodedData) {
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
      ((Chainable) prediction).backpropagate(
          predictionsGradient(outerFunctionGradient[0], prediction.getData(),
              oneHotEncodedLabels.getData()));
    }

    if (oneHotEncodedLabels instanceof Chainable) {
      // We could simply ignore chaining for label, but let's fail fast to allow detecting misuses
      // and eventually change the code in the future
      throw new RuntimeException("CategoricalCrossentropy Labels are not expected to be Variable.");
    }
  }

  protected static double[] predictionsGradient(double outerFunctionGradient, double[] predictionData,
      double[] oneHotEncodedLabelsData) {

    // CrossEntropy = − ∑ i=[0..length] (oneHotLabel[i] * log(pred[i]))
    // ∂CrossEntropy / ∂pred[i] = − (oneHotLabel[i] / pred[i])

    double[] innerGradient = new double[predictionData.length];
    for (int k = 0; k < innerGradient.length; k++) {

      double p = predictionData[k % predictionData.length];
      // As we are clamping the prediction value, the derivate varies only within this range
      if (p > EPSILON && p < 1.0 - EPSILON) {
        innerGradient[k] =
            -outerFunctionGradient * oneHotEncodedLabelsData[k % oneHotEncodedLabelsData.length]
                / predictionData[k % predictionData.length];
      } else {
        innerGradient[k] = 0.0;
      }
    }
    return innerGradient;
  }
}
