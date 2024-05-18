package net.cristcost.differentiable;

import java.util.Arrays;

class MatMulNDimensions {

  static double[] matmul(Tensor tensor, Tensor other) {
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
      Tensor.incrementIndices(indices, resultShape);
    }

    return resultData;
  }

  static int[] matmulShape(Tensor tensor, Tensor other) {
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
        other.getShape().length - 1);

    double sum = 0.0;
    for (int k = 0; k < tensor.getShape()[tensor.getShape().length - 1]; k++) {
      thisIndices[thisIndices.length - 1] = k;
      otherIndices[0] = k;
      sum += tensor.get(thisIndices) * other.get(otherIndices);
    }
    return sum;
  }


}
