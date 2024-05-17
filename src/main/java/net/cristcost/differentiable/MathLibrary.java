package net.cristcost.differentiable;

import java.util.Arrays;

/**
 * A Math Library performing differentiable operation.
 */
public class MathLibrary {

  // Basic operations
  public static Tensor sum(Tensor... operands) {
    double[] result = MathOperationsImplementation.sum(operands);
    return new ConstantTensor(result, operands[0].getShape().clone());
  }

  public static Tensor multiply(Tensor... operands) {
    double[] result = MathOperationsImplementation.multiply(operands);
    return new ConstantTensor(result, operands[0].getShape().clone());
  }

  public static Tensor pow(Tensor base, Tensor exponent) {
    double[] result = MathOperationsImplementation.pow(base, exponent);
    return new ConstantTensor(result, base.getShape().clone());
  }

  public static Tensor relu(Tensor operand) {
    double[] result = MathOperationsImplementation.relu(operand);
    return new ConstantTensor(result, operand.getShape().clone());
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
