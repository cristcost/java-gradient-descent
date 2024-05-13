package net.cristcost.differentiable;

/**
 * A Math Library performing differentiable operation.
 */
public class MathLibrary {

  // Basic operations
  public static ComputedScalar sum(Scalar... operands) {
    return Operation.ADDITION.compute(operands);
  }

  public static ComputedScalar multiply(Scalar... operands) {
    return Operation.MULTIPLICATION.compute(operands);
  }

  public static ComputedScalar pow(Scalar base, Scalar exponent) {
    return Operation.ESPONENTIATION.compute(base, exponent);
  }

  public static ComputedScalar relu(Scalar operand) {
    return Operation.RELU.compute(operand);
  }

  // Types construction
  public static ConstantScalar constant(double value) {
    return new ConstantScalar(value);
  }

}
