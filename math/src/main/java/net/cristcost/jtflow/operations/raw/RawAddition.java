package net.cristcost.jtflow.operations.raw;

public class RawAddition {

  public static double[] compute(int size, double[]... operands) {

    double[] data = new double[size];

    for (double[] operand : operands) {
      if (size % operand.length != 0) {
        throw new IllegalArgumentException("Shapes do not match nor are broadcastable");
      }
      for (int i = 0; i < size; i++) {
        data[i] += operand[i % operand.length];
      }
    }
    return data;

  }

  public static double[] gradient(double[] outerFunctionGradient) {
    return outerFunctionGradient;
  }

}
