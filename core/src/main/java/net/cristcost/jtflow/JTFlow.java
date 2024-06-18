package net.cristcost.jtflow;

import java.util.Arrays;
import net.cristcost.jtflow.api.Broadcastable;
import net.cristcost.jtflow.api.Tensor;
import net.cristcost.jtflow.builder.TensorBuilder;
import net.cristcost.jtflow.tensors.ComputedTensor;
import net.cristcost.jtflow.tensors.ConstantTensor;

/**
 * A Math Library performing differentiable operation.
 */
public class JTFlow {

  // Basic operations
  public static ComputedTensor sum(Tensor... operands) {
    return Operations.ADDITION.compute(operands);
  }

  public static ComputedTensor multiply(Tensor... operands) {
    return Operations.MULTIPLICATION.compute(operands);

  }

  public static ComputedTensor pow(Tensor base, Tensor exponent) {
    return Operations.ESPONENTIATION.compute(base, exponent);
  }

  public static ComputedTensor relu(Tensor operand) {
    return Operations.RELU.compute(operand);
  }

  public static ComputedTensor dot(Tensor a, Tensor b) {
    return Operations.DOT.compute(a, b);
  }

  public static ComputedTensor mse(Tensor a, Tensor b) {
    return Operations.MSE.compute(a, b);
  }

  public static ComputedTensor categoricalCrossentropy(Tensor predictions,
      Tensor oneHotEncodedLabels) {
    return Operations.CATEGORICAL_CROSSENTROPY.compute(predictions, oneHotEncodedLabels);
  }

  public static ComputedTensor softmax(Tensor operand) {
    return Operations.SOFTMAX.compute(operand);
  }

  public static ComputedTensor matmul(Tensor input, Tensor other) {
    return Operations.MATMUL.compute(input, other);
  }

  // Don't like it here, should really be in the tensor only
  public static Tensor broadCast(int[] shape, Tensor operand) {
    if (!Arrays.equals(shape, operand.getShape()) && operand instanceof Broadcastable) {
      return ((Broadcastable) operand).broadcast(shape);
    } else {
      return operand;
    }
  }

  // Don't like it here, should really be in the tensor only
  public static Tensor[] broadCast(int[] shape, Tensor... operands) {
    Tensor[] broadcastOperands = new Tensor[operands.length];
    for (int i = 0; i < operands.length; i++) {
      broadcastOperands[i] = broadCast(shape, operands[i]);
    }
    return broadcastOperands;
  }

  // Don't like it here, should really be in the tensor only
  public static Tensor unsqueeze(int newDimension, Tensor operand) {
    if (operand instanceof Broadcastable) {
      int[] originalShape = operand.getShape();
      int[] newShape = Arrays.copyOf(originalShape, originalShape.length + 1);
      if (newDimension < 0) {
        newDimension += newShape.length;
      }
      for (int i = newShape.length - 1; i > newDimension; i--) {
        newShape[i] = newShape[i - 1];
      }
      newShape[newDimension] = 1;

      return ((Broadcastable) operand).broadcast(newShape);
    } else {
      return operand;
    }
  }

  // Don't like it here, should really be in the tensor only
  public static Tensor[] unsqueeze(int newDimension, Tensor... operands) {
    Tensor[] broadcastOperands = new Tensor[operands.length];
    for (int i = 0; i < operands.length; i++) {
      broadcastOperands[i] = unsqueeze(newDimension, operands[i]);
    }
    return broadcastOperands;
  }

  public static Tensor transpose(Tensor tensor) {
    int rows = tensor.getShape()[0];
    int cols = tensor.getShape()[1];

    double[] matrix = tensor.getData();
    double[] transposed = new double[tensor.size()];

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        transposed[j * rows + i] = matrix[i * cols + j];
      }
    }

    return new ConstantTensor(transposed, new int[] {cols, rows});
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
