package net.cristcost.jtflow.operations.raw;

import java.util.Arrays;

public class RawMultiplication {

  public static double[] compute(int size, double[]... operands) {

    double[] data = new double[size];
    Arrays.fill(data, 1.0);

    for (double[] operand : operands) {
      if (size % operand.length != 0) {
        throw new IllegalArgumentException("Shapes do not match nor are broadcastable");
      }
      for (int i = 0; i < size; i++) {
        data[i] *= operand[i % operand.length];
      }
    }
    return data;

  }

  public static double[] gradient(double[] outerFunctionGradient, int index, double[]... operands) {
    double[] innerGradient = outerFunctionGradient.clone();
    for (int j = 0; j < operands.length; j++) {
      if (index != j) {
        for (int k = 0; k < innerGradient.length; k++) {
          innerGradient[k] *= operands[j][k % operands[j].length];
        }
      }
    }
    return innerGradient;
  }
}
