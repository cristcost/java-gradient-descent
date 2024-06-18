package net.cristcost.differentiable;

class MatMulBackpropagationTest {


  public static void main(String[] args) {

    // VariableTensor x61 = matrix(3, 2).variable().withData(2.0, 4.0, 6.0, 8.0, 10.0, 12.0);
    // VariableTensor x62 = matrix(2, 3).variable().withData(1.0, 3.0, 5.0, 7.0, 9.0, 11.0);
    // ComputedTensor y6 = matmul(x61, x62);
    // assertTensorsEquals(matrix(3, 3).withData(30.0, 42.0, 54.0, 62.0, 90.0, 118.0, 94.0, 138.0,
    // 182.0), y6);
    // y6.backpropagate(matrix(3, 3).ones().getData());
    // System.out.println(Arrays.toString(x61.getGradient()));
    // System.out.println(Arrays.toString(x62.getGradient()));
    // assertArrayEquals(data(9.0, 27.0, 9.0, 27.0, 9.0, 27.0), x61.getGradient());
    // assertArrayEquals(data(18.0, 18.0, 18.0, 24.0, 24.0, 24.0), x62.getGradient());

    double[][] a = {
        {2.0, 4.0},
        {6.0, 8.0},
        {10.0, 12.0}
    };
    double[][] b = {
        {1.0, 3.0, 5.0},
        {7.0, 9.0, 11.0}
    };
    double[][] xBackpropagation = {
        {1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0}
    };

    double[][][] gradients = matmulBackpropagation(a, b, xBackpropagation);

    // Print gradients
    System.out.println("Gradient with respect to a:");
    printMatrix(gradients[0]);

    System.out.println("Gradient with respect to b:");
    printMatrix(gradients[1]);
  }

  public static double[][] matmul(double[][] A, double[][] B) {
    int m = A.length;
    int n = A[0].length;
    int p = B[0].length;
    double[][] result = new double[m][p];

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < p; j++) {
        for (int k = 0; k < n; k++) {
          result[i][j] += A[i][k] * B[k][j];
        }
      }
    }

    return result;
  }

  public static double[][][] matmulBackpropagation(double[][] a, double[][] b,
      double[][] xBackpropagation) {
    double[][] bTransposed = transpose(b);
    double[][] aTransposed = transpose(a);

    double[][] dA = matmul(xBackpropagation, bTransposed);
    double[][] dB = matmul(aTransposed, xBackpropagation);

    return new double[][][] {dA, dB};
  }

  public static double[][] transpose(double[][] matrix) {
    int rows = matrix.length;
    int cols = matrix[0].length;
    double[][] transposed = new double[cols][rows];

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        transposed[j][i] = matrix[i][j];
      }
    }

    return transposed;
  }

  private static void printMatrix(double[][] matrix) {
    for (double[] row : matrix) {
      for (double val : row) {
        System.out.printf("%.6f ", val);
      }
      System.out.println();
    }
  }
}
