package net.cristcost.jtflow.operations.impl;

import net.cristcost.jtflow.api.Chainable;
import net.cristcost.jtflow.api.Tensor;
import net.cristcost.jtflow.operations.raw.RawNegativeLogLikelihoodLoss;

public class NegativeLogLikelihoodLoss {


  public static void chain(double[] outerFunctionGradient, Tensor prediction,
      Tensor oneHotEncodedLabels) {

    if (prediction instanceof Chainable) {
      ((Chainable) prediction).backpropagate(
          RawNegativeLogLikelihoodLoss.gradient(outerFunctionGradient[0],
              prediction.getData(),
              oneHotEncodedLabels.getData()));
    }

    if (oneHotEncodedLabels instanceof Chainable) {
      // We could simply ignore chaining for label, but let's fail fast to allow detecting misuses
      // and eventually change the code in the future
      throw new RuntimeException("CategoricalCrossentropy Labels are not expected to be Variable.");
    }
  }



  public static double[] compute(Tensor prediction, Tensor oneHotEncodedLabels) {
    validateTensorCompatibility(prediction, oneHotEncodedLabels);

    return Common.makeData(
        RawNegativeLogLikelihoodLoss.compute(prediction.getData(), oneHotEncodedLabels.getData()));
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


}
