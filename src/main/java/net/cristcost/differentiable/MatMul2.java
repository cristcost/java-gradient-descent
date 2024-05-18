package net.cristcost.differentiable;

import java.util.Arrays;

class MatMul2 {
  static double[] matmul(Tensor tensor, Tensor other) {
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
          resultData[i * J + j] += tensor.get(i * K + k) * other.get(k * J + j);
        }
      }
    }

    return resultData;
  }

  static int[] matmulShape(Tensor tensor, Tensor other) {

    validateMatrixCompatibility(tensor, other);

    int[] resultShape = Arrays.copyOf(tensor.getShape(), 2);
    resultShape[1] = other.getShape()[1];
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