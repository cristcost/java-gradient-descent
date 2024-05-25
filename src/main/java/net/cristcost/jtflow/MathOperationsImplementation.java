package net.cristcost.jtflow;

import java.util.Arrays;

class MathOperationsImplementation {

  static double[] sum(Tensor... operands) {
    double[] data = new double[operands[0].size()];

    for (Tensor t : operands) {
      if (data.length != t.size()) {
        throw new IllegalArgumentException("Shapes do not match.");
      }
      for (int i = 0; i < data.length; i++) {
        data[i] += t.get(i);
      }
    }
    return data;
  }

  static double[] multiply(Tensor... operands) {
    double[] data = new double[operands[0].size()];
    Arrays.fill(data, 1.0);

    for (Tensor t : operands) {
      if (data.length != t.size()) {
        throw new IllegalArgumentException("Shapes do not match.");
      }
      for (int i = 0; i < data.length; i++) {
        data[i] *= t.get(i);
      }
    }
    return data;
  }

  static double[] pow(Tensor base, Tensor exponent) {
    if (base.size() != exponent.size()) {
      throw new IllegalArgumentException("Shapes do not match.");
    }

    double[] data = new double[base.size()];
    for (int i = 0; i < data.length; i++) {
      data[i] = Math.pow(base.get(i), exponent.get(i));
    }
    return data;
  }

  static double[] relu(Tensor operand) {
    double[] data = new double[operand.size()];
    for (int i = 0; i < data.length; i++) {
      data[i] = Math.max(0.0, operand.get(i));
    }
    return data;
  }
}
