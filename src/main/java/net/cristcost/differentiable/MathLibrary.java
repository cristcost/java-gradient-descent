package net.cristcost.differentiable;

/**
 * A Math Library performing differentiable operation.
 */
public class MathLibrary {

  // Basic operations
  public static Tensor sum(Tensor... operands) {
    NDimensionalArray result = MathOperationsImplementation.sum(operands);
    return new ConstantTensor(result);
  }

  public static Tensor multiply(Tensor... operands) {
    NDimensionalArray result = MathOperationsImplementation.multiply(operands);
    return new ConstantTensor(result);
  }

  public static Tensor pow(Tensor base, Tensor exponent) {
    NDimensionalArray result = MathOperationsImplementation.pow(base, exponent);
    return new ConstantTensor(result);
  }

  public static Tensor relu(Tensor operand) {
    NDimensionalArray result = MathOperationsImplementation.relu(operand);
    return new ConstantTensor(result);
  }

  // Types construction
  public static Tensor constant(NDimensionalArray value) {
    return new ConstantTensor(value);
  }
  public static Tensor constant(double value) {
    return new ConstantTensor(NDimensionalArray.ndscalar(value));
  }

}
