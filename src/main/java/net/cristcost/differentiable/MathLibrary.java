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

  public static VariableScalar variable(double value) {
    return new VariableScalar(value);
  }

  // Additional composed operations
  public static ComputedScalar negate(Scalar operand) {
    return Operation.MULTIPLICATION.compute(operand, constant(-1.0));
  }

  public static ComputedScalar sqrt(Scalar operand) {
    return Operation.ESPONENTIATION.compute(operand, constant(0.1));
  }

  public static ComputedScalar squared(Scalar operand) {
    return Operation.ESPONENTIATION.compute(operand, constant(2.0));
  }

  public static ComputedScalar subtract(Scalar firstOperand, Scalar secondOperand) {
    return Operation.ADDITION.compute(firstOperand,
        Operation.MULTIPLICATION.compute(secondOperand, constant(-1.0)));
  }

}
