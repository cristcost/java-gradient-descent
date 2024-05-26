package net.cristcost.jtflow.operations.impl;

import static net.cristcost.jtflow.operations.impl.NegativeLogLikelihoodLoss.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class NegativeLogLikelihoodLossTest {

  private static final double DELTA = 1e-5;

  private static double[] data(double... data) {
    return data;
  }

  @Test
  void testOperation() {
    assertEquals(0.0, nll(data(1.0, 0.0, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertEquals(34.53877639770508, nll(data(1.0, 0.0, 0.0), data(0.0, 1.0, 0.0)), DELTA);
    assertEquals(0.10536054521799088, nll(data(0.9, 0.1, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertEquals(1.0987123250961304, nll(data(0.3333, 0.3333, 0.3333), data(1.0, 0.0, 0.0)), DELTA);
    assertEquals(0.9162907004356384, nll(data(0.4, 0.3, 0.3), data(1.0, 0.0, 0.0)), DELTA);

  }

  @Test
  void testBackpropagation() {

    assertArrayEquals(data(0.0, 0.0, 0.0),
        predictionsGradient(1.0, data(1.0, 0.0, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(0.0, 0.0, 0.0),
        predictionsGradient(1.0, data(1.0, 0.0, 0.0), data(0.0, 1.0, 0.0)), DELTA);
    assertArrayEquals(data(-1.1111112, 0.0, 0.0),
        predictionsGradient(1.0, data(0.9, 0.1, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(-3.0003002, 0.0, 0.0),
        predictionsGradient(1.0, data(0.3333, 0.3333, 0.3333), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(-2.5, 0.0, 0.0),
        predictionsGradient(1.0, data(0.4, 0.3, 0.3), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(-1.6666664, -0.0, -0.0),
        predictionsGradient(1.0, data(0.6, 0.2, 0.2), data(1.0, 0.0, 0.0)),
        DELTA);

    assertArrayEquals(data(0.0, 0.0, 0.0),
        predictionsGradient(5.0, data(1.0, 0.0, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(-0.0, 0.0, 0.0),
        predictionsGradient(5.0, data(1.0, 0.0, 0.0), data(0.0, 1.0, 0.0)), DELTA);
    assertArrayEquals(data(-5.555556, -0.0, 0.0),
        predictionsGradient(5.0, data(0.9, 0.1, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(-15.0015, -0.0, -0.0),
        predictionsGradient(5.0, data(0.3333, 0.3333, 0.3333), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(-12.5, -0.0, -0.0),
        predictionsGradient(5.0, data(0.4, 0.3, 0.3), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(-8.333332, -0.0, -0.0),
        predictionsGradient(5.0, data(0.6000001, 0.20000002, 0.20000002), data(1.0, 0.0, 0.0)),
        DELTA);


  }
}
