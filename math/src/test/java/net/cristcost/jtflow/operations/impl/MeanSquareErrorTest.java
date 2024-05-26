package net.cristcost.jtflow.operations.impl;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import static net.cristcost.jtflow.operations.impl.MeanSquareError.*;

class MeanSquareErrorTest {
  private static final double DELTA = 1e-5;

  private static double[] data(double... data) {
    return data;
  }

  @Test
  void testOperation() {
    assertEquals(0.0, mse(data(1.0, 0.0, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertEquals(0.6666666865348816, mse(data(1.0, 0.0, 0.0), data(0.0, 1.0, 0.0)), DELTA);
    assertEquals(0.006666668225079775, mse(data(0.9, 0.1, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertEquals(0.2222222089767456, mse(data(0.3333, 0.3333, 0.3333), data(1.0, 0.0, 0.0)), DELTA);
    assertEquals(0.18000000715255737, mse(data(0.4, 0.3, 0.3), data(1.0, 0.0, 0.0)), DELTA);
    assertEquals(0.07999998331069946,
        mse(data(0.6000001, 0.20000002, 0.20000002), data(1.0, 0.0, 0.0)), DELTA);

  }

  @Test
  void testBackpropagation() {

    assertArrayEquals(data(0.0, 0.0, 0.0),
        operandGradient(1.0, data(1.0, 0.0, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(0.0, 0.0, 0.0),
        operandGradient(1.0, data(1.0, 0.0, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(0.6666667, -0.6666667, 0.0),
        operandGradient(1.0, data(1.0, 0.0, 0.0), data(0.0, 1.0, 0.0)), DELTA);
    assertArrayEquals(data(-0.6666667, 0.6666667, 0.0),
        operandGradient(1.0, data(0.0, 1.0, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(-0.066666685, 0.06666667, 0.0),
        operandGradient(1.0, data(0.9, 0.1, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(0.066666685, -0.06666667, 0.0),
        operandGradient(1.0, data(1.0, 0.0, 0.0), data(0.9, 0.1, 0.0)), DELTA);
    assertArrayEquals(data(-0.44446668, 0.2222, 0.2222),
        operandGradient(1.0, data(0.3333, 0.3333, 0.3333), data(1.0, 0.0, 0.0)),
        DELTA);
    assertArrayEquals(data(0.44446668, -0.2222, -0.2222),
        operandGradient(1.0, data(1.0, 0.0, 0.0), data(0.3333, 0.3333, 0.3333)),
        DELTA);
    assertArrayEquals(data(-0.40000004, 0.20000002, 0.20000002),
        operandGradient(1.0, data(0.4, 0.3, 0.3), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(0.40000004, -0.20000002, -0.20000002),
        operandGradient(1.0, data(1.0, 0.0, 0.0), data(0.4, 0.3, 0.3)), DELTA);
    assertArrayEquals(data(-0.26666662, 0.13333336, 0.13333336), operandGradient(
        1.0, data(0.6000001, 0.20000002, 0.20000002), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(0.26666662, -0.13333336, -0.13333336),
        operandGradient(1.0, data(1.0, 0.0, 0.0), data(0.6000001, 0.20000002, 0.20000002)), DELTA);

    assertArrayEquals(data(0.0, 0.0, 0.0),
        operandGradient(5.0, data(1.0, 0.0, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(0.0, 0.0, 0.0),
        operandGradient(5.0, data(1.0, 0.0, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(3.3333335, -3.3333335, 0.0),
        operandGradient(5.0, data(1.0, 0.0, 0.0), data(0.0, 1.0, 0.0)), DELTA);
    assertArrayEquals(data(-3.3333335, 3.3333335, 0.0),
        operandGradient(5.0, data(0.0, 1.0, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(-0.33333343, 0.33333334, 0.0),
        operandGradient(5.0, data(0.9, 0.1, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(0.33333343, -0.33333334, 0.0),
        operandGradient(5.0, data(1.0, 0.0, 0.0), data(0.9, 0.1, 0.0)), DELTA);
    assertArrayEquals(data(-2.2223334, 1.1110001, 1.1110001),
        operandGradient(5.0, data(0.3333, 0.3333, 0.3333), data(1.0, 0.0, 0.0)),
        DELTA);
    assertArrayEquals(data(2.2223334, -1.1110001, -1.1110001),
        operandGradient(5.0, data(1.0, 0.0, 0.0), data(0.3333, 0.3333, 0.3333)),
        DELTA);
    assertArrayEquals(data(-2.0000002, 1.0000001, 1.0000001),
        operandGradient(5.0, data(0.4, 0.3, 0.3), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(2.0000002, -1.0000001, -1.0000001),
        operandGradient(5.0, data(1.0, 0.0, 0.0), data(0.4, 0.3, 0.3)), DELTA);
    assertArrayEquals(data(-1.3333331, 0.66666675, 0.66666675), operandGradient(5.0,
        data(0.6000001, 0.20000002, 0.20000002), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(1.3333331, -0.66666675, -0.66666675),
        operandGradient(5.0, data(1.0, 0.0, 0.0), data(0.6000001, 0.20000002, 0.20000002)), DELTA);
  }
}
