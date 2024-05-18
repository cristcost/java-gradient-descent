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

  public static Tensor scalar(double value) {
    return new ConstantTensor(data(value), shape());
  }

  public static Tensor vector(double... value) {
    return new ConstantTensor(value, shape(value.length));
  }

  public static TensorBuilder matrix(int rows, int columns) {

    return new TensorBuilder(shape(rows, columns), (d, s) -> new ConstantTensor(d, s));
  }

  public static TensorBuilder constant(int... shape) {
    return new TensorBuilder(shape, (d, s) -> new ConstantTensor(d, s));
  }

}
