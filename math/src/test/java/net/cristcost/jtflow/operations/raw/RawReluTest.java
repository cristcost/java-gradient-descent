package net.cristcost.jtflow.operations.raw;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class RawReluTest {

  private static final double DELTA = 1e-5;


  private static double[] data(double... data) {
    return data;
  }

  @Test
  void testOperation() {
    assertArrayEquals(data(0.0, 0.0, 1.0, 0.0, 0.0),
        RawRelu.compute(data(-1.0, 0.0, 1.0, 0.0, -1.0)), DELTA);
  }

  @Test
  void testBackpropagation() {

    assertArrayEquals(data(0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        RawRelu.gradient(data(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            data(-1.0, -0.0001, 0.0001, 1.0, 0.0001, -0.0001, -1.0)),
        DELTA);


    assertArrayEquals(data(0.0, 0.0, 9.0, 0.0, -1.0, 0.0, 0.0),
        RawRelu.gradient(data(3.0, 5.0, 9.0, 0.0, -1.0, -3.0, -5.0),
            data(-1.0, -0.0001, 0.0001, 1.0, 0.0001, -0.0001, -1.0)),
        DELTA);
  }
}
