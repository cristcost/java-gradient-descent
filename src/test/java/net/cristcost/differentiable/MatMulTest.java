package net.cristcost.differentiable;

import static net.cristcost.differentiable.MathLibrary.*;
import static net.cristcost.differentiable.TensorAsserts.assertTensorsEquals;
import org.junit.jupiter.api.Test;

class MatMulTest {

  @Test
  void testScalarProduct() {
    var matrix1By1 = matrix(1, 1);

    // 2 dimensional matrices
    assertTensorsEquals(
        matrix1By1.withData(10.0),
        matmul(matrix1By1.withData(2.0), matrix1By1.withData(5.0)));

    // N dimensional arrays
    assertTensorsEquals(
        matrix1By1.withData(10.0),
        matmulNdim(matrix1By1.withData(2.0), matrix1By1.withData(5.0)));
  }

  @Test
  void testDotProduct() {
    var matrix1By1 = matrix(1, 1);
    var matrix3By1 = matrix(1, 3);
    var matrix1By3 = matrix(3, 1);

    // 2 dimensional matrices
    assertTensorsEquals(
        matrix1By1.withData(12.0),
        matmul(matrix3By1.withData(2.0, 3.0, 4.0), matrix1By3.withData(0.5, -3.0, 5.0)));

    // N dimensional arrays
    assertTensorsEquals(
        matrix1By1.withData(12.0),
        matmulNdim(matrix3By1.withData(2.0, 3.0, 4.0), matrix1By3.withData(0.5, -3.0, 5.0)));
  }

}
