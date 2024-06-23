package net.cristcost.jtflow.operations.impl;

import java.util.Arrays;
import net.cristcost.jtflow.api.Chainable;
import net.cristcost.jtflow.api.Tensor;
import net.cristcost.jtflow.operations.raw.RawMatMul;

public class MatMul {
  public static void chain(double[] outerFunctionGradient, Tensor a, Tensor b) {

    validateMatrixCompatibility(a, b);

    int firstMatrixRows = a.getShape()[0];
    int commonDimension = a.getShape()[1];
    int secondMatrixColumns = b.getShape()[1];

    a.ifChainable(c -> c
        .backpropagate(
            RawMatMul.firstMatrixGradient(outerFunctionGradient, firstMatrixRows, commonDimension,
                secondMatrixColumns, b.getData())));

    b.ifChainable(c -> c
        .backpropagate(
            RawMatMul.secondMatrixGradient(outerFunctionGradient, firstMatrixRows, commonDimension,
                secondMatrixColumns, a.getData())));

  }


  public static double[] compute(Tensor tensor, Tensor other) {
    validateMatrixCompatibility(tensor, other);

    return RawMatMul.compute(tensor.getData(), other.getData(), tensor.getShape()[0],
        tensor.getShape()[1],
        other.getShape()[1]);
  }

  public static int[] shape(Tensor tensor, Tensor other) {

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
          String.format("Matrix dimensions are not compatible for multiplication: %s vs %s",
              Arrays.toString(tensor.getShape()),
              Arrays.toString(other.getShape())));
    }
  }
}
