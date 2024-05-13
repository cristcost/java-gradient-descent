package net.cristcost.differentiable;

class MathOperationsImplementation {

  static double sum(Scalar... operands) {
    double result = 0.0;
    for (Scalar s : operands) {
      result += s.getValue();
    }
    return result;
  }

  static double multiply(Scalar... operands) {
    double result = 1.0;
    for (Scalar s : operands) {
      result *= s.getValue();
    }
    return result;
  }

  static double pow(Scalar base, Scalar exponent) {
    return Math.pow(base.getValue(), exponent.getValue());
  }

  static double relu(Scalar operand) {
    return operand.getValue() >= 0.0 ? operand.getValue() : 0.0;
  }

}
