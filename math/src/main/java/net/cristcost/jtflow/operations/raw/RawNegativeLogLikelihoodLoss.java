package net.cristcost.jtflow.operations.raw;

public class RawNegativeLogLikelihoodLoss {
  private static final double EPSILON = 1e-15;

  public static double compute(double[] predictionData, double[] oneHotEncodedData) {
    double result = 0.0;
    for (int i = 0; i < predictionData.length; i++) {
      result -=
          oneHotEncodedData[i]
              * Math.log(RawCommon.clamp(predictionData[i], EPSILON, 1.0 - EPSILON));
    }
    return result;
  }

  public static double[] gradient(double outerFunctionGradient,
      double[] predictionData,
      double[] oneHotEncodedLabelsData) {

    // CrossEntropy = − ∑ i=[0..length] (oneHotLabel[i] * log(pred[i]))
    // ∂CrossEntropy / ∂pred[i] = − (oneHotLabel[i] / pred[i])

    double[] innerGradient = new double[predictionData.length];
    for (int k = 0; k < innerGradient.length; k++) {

      double p = predictionData[k % predictionData.length];
      // As we are clamping the prediction value, the derivate varies only within this range
      // if (p > EPSILON && p < 1.0 - EPSILON) {
      // looks like pytorch is not clamping the prediction to a max
      if (p > EPSILON) {
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
