package net.cristcost.jtflow.operations.raw;

public class RawMatMul {


  public static double[] compute(double[] firstMatrixData, double[] secondMatrixData,
      int firstMatrixRows, int commonDimension, int secondMatrixColumns) {

    if (firstMatrixData.length != firstMatrixRows * commonDimension) {
      // TODO: We can probably broadcast some dimensions, for now let's expect exact shape
      throw new IllegalArgumentException(
          String.format("First Matrix shape (%d x %d) does not match the data size: %d",
              firstMatrixRows, commonDimension, firstMatrixData.length));
    }
    if (secondMatrixData.length != commonDimension * secondMatrixColumns) {
      // TODO: We can probably broadcast some dimensions, for now let's expect exact shape
      throw new IllegalArgumentException(
          String.format("Second Matrix shape (%d x %d) does not match the data size: %d",
              commonDimension, secondMatrixColumns, secondMatrixData.length));
    }



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


  public static double[] firstMatrixGradient(double[] outerFunctionGradient, int firstMatrixRows,
      int commonDimension, int secondMatrixColumns, double[] secondMatrixData) {
    double[] innerGradient = RawMatMul.compute(
        outerFunctionGradient,
        transpose(secondMatrixData, commonDimension, secondMatrixColumns),
        firstMatrixRows,
        secondMatrixColumns,
        commonDimension);
    return innerGradient;
  }


  public static double[] secondMatrixGradient(double[] outerFunctionGradient, int firstMatrixRows,
      int commonDimension, int secondMatrixColumns, double[] firstMatrixData) {
    double[] innerGradient = RawMatMul.compute(
        transpose(firstMatrixData, firstMatrixRows, commonDimension),
        outerFunctionGradient,
        commonDimension,
        firstMatrixRows,
        secondMatrixColumns);
    return innerGradient;
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
}
