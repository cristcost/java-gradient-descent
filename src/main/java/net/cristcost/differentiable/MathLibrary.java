package net.cristcost.differentiable;

import java.util.Arrays;

/**
 * A Math Library performing differentiable operation.
 */
public class MathLibrary {

  // Basic operations
  public static Tensor sum(Tensor... operands) {
    return Operation.ADDITION.compute(operands);
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


  public static Tensor multiply(Tensor... operands) {
    return Operation.MULTIPLICATION.compute(operands);

  }

  public static Tensor pow(Tensor base, Tensor exponent) {
    return Operation.ESPONENTIATION.compute(base, exponent);
  }

  public static Tensor relu(Tensor operand) {
    return Operation.RELU.compute(operand);
  }


  public static int[] findResultShape(Tensor... operands) {
    int[] fullShape = shape();
    for (Tensor t : operands) {
      if (fullShape.length < t.getShape().length) {
        fullShape = t.getShape();
      }
    }
    return fullShape.clone();
  }


  public static double[] data(double... data) {
    return data;
  }

  public static int[] shape(int... shape) {
    return shape.clone();
  }

  public static Tensor zeros_like(Tensor ref) {
    double[] data = new double[ref.size()];
    Arrays.fill(data, 0.0);
    return new ConstantTensor(data, ref.getShape().clone());
  }

  public static Tensor ones_like(Tensor ref) {
    double[] data = new double[ref.size()];
    Arrays.fill(data, 1.0);
    return new ConstantTensor(data, ref.getShape().clone());
  }


  @Deprecated
  public static ConstantTensor scalar(double value) {
    return scalar().withData(value);
  }

  @Deprecated
  public static ConstantTensor vector(double... value) {
    return vector(value.length).withData(value);
  }

  public static TensorBuilder<ConstantTensor> scalar() {
    return TensorBuilder.constant(shape());
  }

  public static TensorBuilder<ConstantTensor> vector(int lenght) {
    return TensorBuilder.constant(shape(lenght));
  }

  public static TensorBuilder<ConstantTensor> matrix(int rows, int columns) {
    return TensorBuilder.constant(shape(rows, columns));
  }

  public static TensorBuilder<ConstantTensor> tensor(int... shape) {
    return TensorBuilder.constant(shape);
  }

}
