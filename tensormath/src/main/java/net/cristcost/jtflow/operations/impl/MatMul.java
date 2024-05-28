package net.cristcost.jtflow.operations.impl;

import java.util.Arrays;
import net.cristcost.jtflow.api.Chainable;
import net.cristcost.jtflow.api.Tensor;

// Not tested and not to be used in this form
@Deprecated()
public class MatMul {
  public static void chain(double[] outerFunctionGradient, Tensor a, Tensor b) {

    validateMatrixCompatibility(a, b);

    int firstMatrixRows = a.getShape()[0];
    int commonDimension = a.getShape()[1];
    int secondMatrixColumns = b.getShape()[1];

    if (a instanceof Chainable) {
      double[] innerGradient = matmul(
          outerFunctionGradient,
          transpose(b.getData(), commonDimension, secondMatrixColumns),
          firstMatrixRows,
          secondMatrixColumns,
          commonDimension);

      ((Chainable) a).backpropagate(innerGradient);
    }

    if (b instanceof Chainable) {
      double[] innerGradient = matmul(
          transpose(a.getData(), firstMatrixRows, commonDimension),
          outerFunctionGradient,
          commonDimension,
          firstMatrixRows,
          secondMatrixColumns);

      ((Chainable) b).backpropagate(innerGradient);
    }
  }

  public static double[] compute(Tensor tensor, Tensor other) {
    validateMatrixCompatibility(tensor, other);

    return matmul(tensor.getData(), other.getData(), tensor.getShape()[0], tensor.getShape()[1],
        other.getShape()[1]);
  }

  public static int[] shape(Tensor tensor, Tensor other) {

    validateMatrixCompatibility(tensor, other);

    int[] resultShape = Arrays.copyOf(tensor.getShape(), 2);
    resultShape[1] = other.getShape()[1];
    return resultShape;
  }

  private static double[] matmul(double[] firstMatrixData, double[] secondMatrixData,
      int firstMatrixRows, int commonDimension, int secondMatrixColumns) {

    double[] resultData = new double[firstMatrixRows * secondMatrixColumns];

    // c(i,j) = a(i,1) * b(1,j) + ... + a(i,n) * b(n,j)
    for (int row = 0; row < firstMatrixRows; row++) {
      for (int column = 0; column < secondMatrixColumns; column++) {
        for (int common = 0; common < commonDimension; common++) {
          resultData[row * secondMatrixColumns + column] +=
              firstMatrixData[row * commonDimension + common]
                  * secondMatrixData[common * secondMatrixColumns + column];
        }
      }
    }

    return resultData;
  }


  private static double[] transpose(double[] matrix, int rows, int cols) {
    double[] transposed = new double[rows * cols];

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        transposed[j * rows + i] = matrix[i * cols + j];
      }
    }

    return transposed;
  }


  private static void validateMatrixCompatibility(Tensor tensor, Tensor other) {
    // Check if both arrays have at least 2 dimensions
    if (tensor.getShape().length != 2 || other.getShape().length != 2) {
      throw new IllegalArgumentException(
          "Matrix multiplication requires arrays with of dimension 2.");
    }

    if (tensor.getShape()[1] != other.getShape()[0]) {
      throw new IllegalArgumentException(
          String.format("Matrix dimensions are not compatible for multiplication: %s vs %s",
              Arrays.toString(tensor.getShape()),
              Arrays.toString(other.getShape())));
    }
  }
}
