package net.cristcost.differentiable;

import java.util.Arrays;

/**
 * A Math Library performing differentiable operation.
 */
public class MathLibrary {

  // Basic operations
  public static Tensor sum(Tensor... operands) {
    int[] shape = findResultShape(operands);
    double[] result = MathOperationsImplementation.sum(broadCast(shape, operands));
    return new ConstantTensor(result, shape);
  }


  private static Tensor broadCast(int[] shape, Tensor operand) {
    if (!Arrays.equals(shape, operand.getShape()) && operand instanceof Broadcastable) {
      return ((Broadcastable) operand).broadcast(shape);
    } else {
      return operand;
    }
  }

  private static Tensor[] broadCast(int[] shape, Tensor... operands) {
    Tensor[] broadcastOperands = new Tensor[operands.length];
    for (int i = 0; i < operands.length; i++) {
      broadcastOperands[i] = broadCast(shape, operands[i]);
    }
    return broadcastOperands;
  }


  public static Tensor multiply(Tensor... operands) {
    int[] shape = findResultShape(operands);
    double[] result = MathOperationsImplementation.multiply(broadCast(shape, operands));
    return new ConstantTensor(result, shape);
  }

  public static Tensor pow(Tensor base, Tensor exponent) {
    int[] shape = findResultShape(base, exponent);
    double[] result =
        MathOperationsImplementation.pow(broadCast(shape, base), broadCast(shape, exponent));
    return new ConstantTensor(result, shape);
  }

  public static Tensor relu(Tensor operand) {
    double[] result = MathOperationsImplementation.relu(operand);
    return new ConstantTensor(result, operand.getShape().clone());
  }


  private static int[] findResultShape(Tensor... operands) {
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

  public static BuildWithData matrix(int rows, int columns) {
    return data -> new ConstantTensor(data, shape(rows, columns));
  }

  public static BuildWithShape constant(double... data) {
    return shape -> new ConstantTensor(data, shape);
  }

  public interface BuildWithShape {
    public Tensor shape(int... shape);
  }
  public interface BuildWithData {
    public Tensor data(double... data);
  }

}
