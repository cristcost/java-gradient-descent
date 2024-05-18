package net.cristcost.differentiable;

import java.util.Arrays;

/**
 * A Math Library performing differentiable operation.
 */
public class MathLibrary {

  // Basic operations
  public static ComputedTensor sum(Tensor... operands) {
    return Operation.ADDITION.compute(operands);
  }

  public static ComputedTensor multiply(Tensor... operands) {
    return Operation.MULTIPLICATION.compute(operands);

  }

  public static ComputedTensor pow(Tensor base, Tensor exponent) {
    return Operation.ESPONENTIATION.compute(base, exponent);
  }

  public static ComputedTensor relu(Tensor operand) {
    return Operation.RELU.compute(operand);
  }

  public static ComputedTensor matmul(Tensor input, Tensor other) {
    return Operation.MATMUL.compute(input, other);
  }

  public static ComputedTensor matmulNdim(Tensor input, Tensor other) {
    return Operation.MATMUL_NDIM.compute(input, other);
  }

  private static Tensor broadCast(int[] shape, Tensor operand) {
    if (!Arrays.equals(shape, operand.getShape()) && operand instanceof Broadcastable) {
      return ((Broadcastable) operand).broadcast(shape);
    } else {
      return operand;
    }
  }

  public static Tensor[] broadCast(int[] shape, Tensor... operands) {
    Tensor[] broadcastOperands = new Tensor[operands.length];
    for (int i = 0; i < operands.length; i++) {
      broadcastOperands[i] = broadCast(shape, operands[i]);
    }
    return broadcastOperands;
  }

  public static double[] data(double... data) {
    return data;
  }

  public static int[] shape(int... shape) {
    return shape.clone();
  }

  public static int[] index(int... indices) {
    return indices.clone();
  }

  public static ConstantTensor eye(int n) {
    var squareMatrix = TensorBuilder.builder(shape(n, n));
    return squareMatrix.withData((i, j) -> i == j ? 1.0 : 0.0);
  }

  public static ConstantTensor scalar(double value) {
    return scalar().withData(value);
  }

  public static ConstantTensor vector(double... values) {
    return vector(values.length).withData(values);
  }

  public static TensorBuilder<ConstantTensor> scalar() {
    return TensorBuilder.builder(shape());
  }

  public static TensorBuilder<ConstantTensor> vector(int lenght) {
    return TensorBuilder.builder(shape(lenght));
  }

  public static TensorBuilder<ConstantTensor> matrix(int rows, int columns) {
    return TensorBuilder.builder(shape(rows, columns));
  }


  public static TensorBuilder<ConstantTensor> tensor(int... shape) {
    return TensorBuilder.builder(shape);
  }

  public static TensorBuilder<ConstantTensor> like(Tensor ref) {
    return TensorBuilder.builder(ref.getShape());
  }

}
