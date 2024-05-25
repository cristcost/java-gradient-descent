package net.cristcost.differentiable;

import static net.cristcost.differentiable.TensorAsserts.assertTensorsEquals;
import static net.cristcost.jtflow.JTFlow.*;
import org.junit.jupiter.api.Test;

class MatMulTest {

  @Test
  void testScalarProduct() {
    var matrix1By1 = matrix(1, 1);

    // 2 dimensional matrices
    assertTensorsEquals(
        matrix1By1.withData(10.0),
        matmul(matrix1By1.withData(2.0), matrix1By1.withData(5.0)));
  }

  @Test
  void testDotProduct() {
    var matrix1By1 = matrix(1, 1);
    var matrix1By3 = matrix(1, 3);
    var matrix3By1 = matrix(3, 1);

    // 2 dimensional matrices
    assertTensorsEquals(
        matrix1By1.withData(12.0),
        matmul(matrix1By3.withData(2.0, 3.0, 4.0), matrix3By1.withData(0.5, -3.0, 5.0)));
  }

  @Test
  void testMatmul() {
    var matrix3By3 = matrix(3, 3);


    // 2 dimensional matrices
    assertTensorsEquals(
        matrix3By3.withData(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
        matmul(
            matrix3By3.withData(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
            eye(3)));

    assertTensorsEquals(
        matrix3By3.zeros(),
        matmul(
            matrix3By3.withData(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
            matrix3By3.zeros()));

    assertTensorsEquals(
        multiply(matrix3By3.ones(), scalar(3.0)),
        matmul(
            matrix3By3.ones(),
            matrix3By3.ones()));

    assertTensorsEquals(
        matrix3By3.withData(-5.0, 2.0, 11.0, -17.0, 2.0, 23.0, -29.0, 2.0, 35.0),
        matmul(
            matrix3By3.withData(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
            matrix3By3.withData(-3.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 3.0)));

  }


}
