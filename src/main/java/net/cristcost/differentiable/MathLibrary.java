package net.cristcost.differentiable;

/**
 * A Math Library performing differentiable operation.
 */
public class MathLibrary {

  // Basic operations
  public static Scalar sum(Scalar... operands) {
    double result = MathOperationsImplementation.sum(operands);
    return new ConstantScalar(result);
  }

  public static Scalar multiply(Scalar... operands) {
    double result = MathOperationsImplementation.multiply(operands);
    return new ConstantScalar(result);
  }

  public static Scalar pow(Scalar base, Scalar exponent) {
    double result = MathOperationsImplementation.pow(base, exponent);
    return new ConstantScalar(result);
  }

  public static Scalar relu(Scalar operand) {
    double result = MathOperationsImplementation.relu(operand);
    return new ConstantScalar(result);
  }

  // Types construction
  public static Scalar constant(double value) {
    return new ConstantScalar(value);
  }

}
