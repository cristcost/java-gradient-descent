package net.cristcost.jtflow.operations.raw;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class RawMatMulTest {


  private static final double DELTA = 1e-5;


  private static double[] data(double... data) {
    return data;
  }

  @Test
  void testOperation() {

    assertArrayEquals(data(2.0),
        RawMatMul.compute(data(1.0, 1.0), data(1.0, 1.0), 1, 2, 1),
        DELTA);

    assertArrayEquals(data(2.0, 2.0),
        RawMatMul.compute(data(1.0, 1.0, 1.0, 1.0), data(1.0, 1.0), 2, 2, 1),
        DELTA);
    assertArrayEquals(data(2.0, 2.0),
        RawMatMul.compute(data(1.0, 1.0), data(1.0, 1.0, 1.0, 1.0), 1, 2, 2),
        DELTA);
    assertArrayEquals(data(2.0, 2.0, 2.0, 2.0),
        RawMatMul.compute(data(1.0, 1.0, 1.0, 1.0), data(1.0, 1.0, 1.0, 1.0), 2, 2, 2),
        DELTA);

    assertArrayEquals(data(1.0, 1.0, 1.0, 1.0),
        RawMatMul.compute(data(1.0, 1.0), data(1.0, 1.0), 2, 1, 2),
        DELTA);


    assertThrows(RuntimeException.class, () -> {
      RawMatMul.compute(data(1.0, 1.0, 1.0), data(1.0, 1.0), 1, 2, 1);
    });
    assertThrows(RuntimeException.class, () -> {
      RawMatMul.compute(data(1.0, 1.0), data(1.0, 1.0, 1.0), 1, 2, 1);
    });
  }

  @Test
  void testBackpropagation() {

    assertArrayEquals(data(2.0),
        RawMatMul.firstMatrixGradient(data(1.0), 1, 1, 1, data(2.0)), DELTA);

    assertArrayEquals(data(2.0),
        RawMatMul.secondMatrixGradient(data(1.0), 1, 1, 1, data(2.0)), DELTA);


    assertArrayEquals(data(1.0, 2.0, 3.0),
        RawMatMul.firstMatrixGradient(data(1.0), 1, 3, 1, data(1.0, 2.0, 3.0)), DELTA);

    assertArrayEquals(data(1.0, 2.0, 3.0),
        RawMatMul.secondMatrixGradient(data(1.0), 1, 3, 1, data(1.0, 2.0, 3.0)), DELTA);


    assertArrayEquals(data(3.0, 7.0, 11.0, 3.0, 7.0, 11.0, 3.0, 7.0, 11.0),
        RawMatMul.firstMatrixGradient(data(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), 3, 3, 2,
            data(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
        DELTA);

    assertArrayEquals(data(12.0, 12.0, 15.0, 15.0, 18.0, 18.0),
        RawMatMul.secondMatrixGradient(data(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), 3, 3, 2,
            data(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)),
        DELTA);

  }

}
