package net.cristcost.jtflow.operations.raw;

public class RawRelu {

  public static double[] compute(double[] operand) {
    double[] data = new double[operand.length];
    for (int i = 0; i < data.length; i++) {
      data[i] = Math.max(0.0, operand[i]);
    }
    return data;

  }

  public static double[] gradient(double[] outerFunctionGradient, double[] operand) {
    double[] innerGradient = new double[outerFunctionGradient.length];
    for (int k = 0; k < innerGradient.length; k++) {
      innerGradient[k] = operand[k % operand.length] > 0.0 ? outerFunctionGradient[k] : 0.0;
    }
    return innerGradient;
  }

}
