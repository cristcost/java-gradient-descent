package net.cristcost.differentiable;

import java.util.Arrays;

class MathOperationsImplementation {

  static double[] sum(Tensor... operands) {
    double[] data = new double[operands[0].size()];

    for (Tensor t : operands) {
      if (data.length != t.size()) {
        throw new IllegalArgumentException("Shapes do not match.");
      }
      for (int i = 0; i < data.length; i++) {
        data[i] += t.get(i);
      }
    }
    return data;
  }

  static double[] multiply(Tensor... operands) {
    double[] data = new double[operands[0].size()];
    Arrays.fill(data, 1.0);

    for (Tensor t : operands) {
      if (data.length != t.size()) {
        throw new IllegalArgumentException("Shapes do not match.");
      }
      for (int i = 0; i < data.length; i++) {
        data[i] *= t.get(i);
      }
    }
    return data;
  }

  static double[] pow(Tensor base, Tensor exponent) {
    if (base.size() != exponent.size()) {
      throw new IllegalArgumentException("Shapes do not match.");
    }

    double[] data = new double[base.size()];
    for (int i = 0; i < data.length; i++) {
      data[i] = Math.pow(base.get(i), exponent.get(i));
    }
    return data;
  }

  static double[] relu(Tensor operand) {
    double[] data = new double[operand.size()];
    for (int i = 0; i < data.length; i++) {
      data[i] = Math.max(0.0, operand.get(i));
    }
    return data;
  }

  public double[] matmul(Tensor tensor, Tensor other) {
    return MatMul2.matmul(tensor, other);
  }

  private static class MatMul2 {
    public static double[] matmul(Tensor tensor, Tensor other) {
      validateMatrixCompatibility(tensor, other);

      double[] resultData = new double[tensor.getShape()[0] * other.getShape()[1]];

      int K = tensor.getShape()[1];
      int I = tensor.getShape()[0];
      int J = other.getShape()[1];

      // c(i,j) = a(i,1) * b(1,j) + ... + a(i,n) * b(n,j)
      for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
          for (int k = 0; k < K; k++) {
            // resultData[i * J + j] += tensor.get(i, k) * other.get(k, j);
            resultData[i * J + j] += tensor.get(i * K + k) * other.get(k * K + j);
          }
        }
      }
      
      return resultData;
    }

    public static int[] matmulShape(Tensor tensor, Tensor other) {

      validateMatrixCompatibility(tensor, other);

      int[] resultShape = Arrays.copyOf(tensor.getShape(), 3);
      resultShape[2] = other.getShape()[1];
      return resultShape;
    }

    private static void validateMatrixCompatibility(Tensor tensor, Tensor other) {
      // Check if both arrays have at least 2 dimensions
      if (tensor.getShape().length != 2 || other.getShape().length != 2) {
        throw new IllegalArgumentException(
            "Matrix multiplication requires arrays with of dimension 2.");
      }

      if (tensor.getShape()[1] != other.getShape()[0]) {
        throw new IllegalArgumentException(
            "Matrix dimensions are not compatible for multiplication.");
      }
    }

  }

  private static class MatMulNDimensions {

    public static double[] matmul(Tensor tensor, Tensor other) {
      // Check if both arrays have at least 2 dimensions
      if (tensor.getShape().length < 2 || other.getShape().length < 2) {
        throw new IllegalArgumentException(
            "Matrix multiplication requires arrays with at least 2 dimensions.");
      }

      // Check if the inner dimensions are compatible for multiplication
      if (tensor.getShape()[tensor.getShape().length - 1] != other.getShape()[0]) {
        throw new IllegalArgumentException(
            "Matrix dimensions are not compatible for multiplication.");
      }

      int[] resultShape = matmulShape(tensor, other);

      // Initialize result array
      int resultSize = Arrays.stream(resultShape).reduce(1, (a, b) -> a * b);
      double[] resultData = new double[resultSize];

      // Perform matrix multiplication
      int[] indices = new int[tensor.getShape().length + other.getShape().length - 2];
      for (int i = 0; i < resultData.length; i++) {
        resultData[i] = computeElement(indices, tensor, other);
        incrementIndices(indices, resultShape);
      }

      return resultData;
    }

    public static int[] matmulShape(Tensor tensor, Tensor other) {
      // Compute the shape of the result array
      int[] resultShape = new int[tensor.getShape().length + other.getShape().length - 2];
      System.arraycopy(tensor.getShape(), 0, resultShape, 0, tensor.getShape().length - 1);
      System.arraycopy(other.getShape(), 1, resultShape, other.getShape().length - 1,
          other.getShape().length - 1);
      return resultShape;
    }

    private static double computeElement(int[] indices, Tensor tensor, Tensor other) {
      int[] thisIndices = new int[tensor.getShape().length];
      int[] otherIndices = new int[other.getShape().length];
      System.arraycopy(indices, 0, thisIndices, 0, tensor.getShape().length - 1);
      System.arraycopy(indices, tensor.getShape().length - 1, otherIndices, 1,
          other.getShape().length - 2);

      double sum = 0.0;
      for (int k = 0; k < tensor.getShape()[tensor.getShape().length - 1]; k++) {
        thisIndices[thisIndices.length - 1] = k;
        otherIndices[0] = k;
        sum += tensor.get(thisIndices) * other.get(otherIndices);
      }
      return sum;
    }

    private static void incrementIndices(int[] indices, int[] shape) {
      for (int i = indices.length - 1; i >= 0; i--) {
        indices[i]++;
        if (indices[i] < shape[i]) {
          break;
        } else {
          indices[i] = 0;
        }
      }
    }
  }


}
