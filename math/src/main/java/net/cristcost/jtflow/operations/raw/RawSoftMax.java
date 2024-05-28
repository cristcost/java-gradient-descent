package net.cristcost.jtflow.operations.raw;

import java.util.Arrays;

public class RawSoftMax {
  public static double[] compute(double[] data) {
    double[] result = new double[data.length];
    double sum = 0.0;

    // this is avoid Numerical instability large exponent numbers, we shift the values used for
    // computation by the max. We can do that as Softmax function is invariant to translation by a
    // constant
    double numericalInstabilityCorrection =
        Arrays.stream(data).max().orElse(Double.NEGATIVE_INFINITY);

    for (int i = 0; i < data.length; i++) {
      result[i] = Math.exp(data[i] - numericalInstabilityCorrection);
      sum += result[i];
    }
    for (int i = 0; i < data.length; i++) {
      result[i] /= sum;
    }

    return result;
  }

  public static double[] gradient(double[] outerFunctionGradient, double[] data) {
    double[] softmax = new double[data.length];
    double[] innerGradient = new double[data.length];
    double sum = 0.0;

    // this is avoid Numerical instability large exponent numbers, we shift the values used for
    // computation by the max. We can do that as Softmax function is invariant to translation by a
    // constant AND chain rule is based on the softmax
    double numericalInstabilityCorrection =
        Arrays.stream(data).max().orElse(Double.NEGATIVE_INFINITY);

    for (int i = 0; i < data.length; i++) {
      softmax[i] = Math.exp(data[i] - numericalInstabilityCorrection);
      sum += softmax[i];
    }

    for (int i = 0; i < innerGradient.length; i++) {
      for (int j = 0; j < innerGradient.length; j++) {
        double delta = (i == j) ? 1.0 : 0.0;
        innerGradient[i] +=
            outerFunctionGradient[j] * softmax[i] / sum * (delta - softmax[j] / sum);
      }
    }
    return innerGradient;
  }
}
