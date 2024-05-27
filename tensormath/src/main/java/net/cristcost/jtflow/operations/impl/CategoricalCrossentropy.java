package net.cristcost.jtflow.operations.impl;

import java.util.Arrays;
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
    return NegativeLogLikelihoodLoss.nll(SoftMax.softmax(predictionData), oneHotEncodedData);
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

      double[] predictionData = prediction.getData();
      double[] oneHotEncodedLabelsData = oneHotEncodedLabels.getData();
      double[] innerGradient =
          predictionsGradient(outerFunctionGradient[0], predictionData, oneHotEncodedLabelsData);

      ((Chainable) prediction).backpropagate(innerGradient);
    }

    if (oneHotEncodedLabels instanceof Chainable) {
      // We could simply ignore chaining for label, but let's fail fast to allow detecting misuses
      // and eventually change the code in the future
      throw new RuntimeException("CategoricalCrossentropy Labels are not expected to be Variable.");
    }
  }

  protected static double[] predictionsGradient(double outerFunctionGradient,
      double[] predictionData,
      double[] oneHotEncodedLabelsData) {


    // f(g(x)) = nll(softmax(x), oneHotEncodedData)
    // ∂f(g(x))/∂x = f'(g(x)) * g'(x) = nll'(softmax(x), oneHotEncodedData) * softmax'(x)

    double[] nllGradient = NegativeLogLikelihoodLoss.predictionsGradient(outerFunctionGradient,
        SoftMax.softmax(predictionData), oneHotEncodedLabelsData);

    double[] gradient = SoftMax.gradient(nllGradient, predictionData);

    return gradient;
  }



}
