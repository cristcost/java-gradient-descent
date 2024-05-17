package net.cristcost.differentiable;

import java.util.Arrays;

class MathOperationsImplementation {

  static NDimensionalArray sum(Tensor... operands) {

    double[] data = new double[operands[0].getValue().getData().length];
    int[] shape =
        Arrays.copyOf(operands[0].getValue().getShape(), operands[0].getValue().getShape().length);

    for (Tensor s : operands) {
      if (!Arrays.equals(shape, s.getValue().getShape())) {
        throw new IllegalArgumentException("Shapes do not match.");
      }
      for (int i = 0; i < data.length; i++) {
        data[i] += s.getValue().getData()[i];
      }
    }
    return new NDimensionalArray(data, shape);
  }

  static NDimensionalArray multiply(Tensor... operands) {

    double[] data = new double[operands[0].getValue().getData().length];
    Arrays.fill(data, 1.0);

    int[] shape =
        Arrays.copyOf(operands[0].getValue().getShape(), operands[0].getValue().getShape().length);

    for (Tensor s : operands) {
      if (!Arrays.equals(shape, s.getValue().getShape())) {
        throw new IllegalArgumentException("Shapes do not match.");
      }
      for (int i = 0; i < data.length; i++) {
        data[i] *= s.getValue().getData()[i];
      }
    }
    return new NDimensionalArray(data, shape);
  }

  static NDimensionalArray pow(Tensor base, Tensor exponent) {
    double[] data = new double[base.getValue().getData().length];
    Arrays.fill(data, 1.0);

    int[] shape =
        Arrays.copyOf(base.getValue().getShape(), base.getValue().getShape().length);

    if (!Arrays.equals(shape, exponent.getValue().getShape())) {
      throw new IllegalArgumentException("Shapes do not match.");
    }
    for (int i = 0; i < data.length; i++) {
      data[i] *=
          Math.pow(
              base.getValue().getData()[i],
              exponent.getValue().getData()[i]);
    }
    return new NDimensionalArray(data, shape);
  }

  static NDimensionalArray relu(Tensor operand) {
    double[] data = new double[operand.getValue().getData().length];
    Arrays.fill(data, 1.0);

    int[] shape =
        Arrays.copyOf(operand.getValue().getShape(), operand.getValue().getShape().length);

    if (!Arrays.equals(shape, operand.getValue().getShape())) {
      throw new IllegalArgumentException("Shapes do not match.");
    }
    for (int i = 0; i < data.length; i++) {
      data[i] = Math.max(0.0, operand.getValue().getData()[i]);
    }
    return new NDimensionalArray(data, shape);
  }

}
