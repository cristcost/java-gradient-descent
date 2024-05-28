package net.cristcost.jtflow.operations.raw;

public class RawMeanSquareError {
  public static double compute(double[] aData, double[] bData) {
    double result = 0.0;

    for (int i = 0; i < aData.length; i++) {
      result += Math.pow(aData[i % aData.length] - bData[i % bData.length], 2.0);
    }
    result /= aData.length;
    return result;
  }

  public static double[] gradient(double outerFunctionGradient,
      double[] variantData,
      double[] invariantData) {
    double[] innerGradient = new double[variantData.length];

    for (int k = 0; k < innerGradient.length; k++) {
      innerGradient[k] = outerFunctionGradient *
          (2 * variantData[k % variantData.length] - 2 * invariantData[k % invariantData.length])
          / innerGradient.length;

    }
    return innerGradient;
  }
}
