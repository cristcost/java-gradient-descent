package net.cristcost.jtflow.operations.raw;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class RawAdditionTest {
  private static final double DELTA = 1e-5;


  private static double[] data(double... data) {
    return data;
  }

  @Test
  void testOperation() {

    assertArrayEquals(data(1.0, 1.0, 1.0, 1.0, 1.0), RawAddition.compute(5, data(1.0)), DELTA);

    assertArrayEquals(data(5.0, 5.0, 5.0, 5.0, 5.0),
        RawAddition.compute(5, data(1.0), data(1.0), data(1.0), data(1.0), data(1.0)), DELTA);

    assertArrayEquals(data(111.1, 122.2, 113.3, 121.4, 112.5, 123.6),
        RawAddition.compute(6,
            data(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            data(1.0, 2.0, 3.0),
            data(10.0, 20.0),
            data(100.0)),
        DELTA);

    assertThrows(RuntimeException.class, () -> {
      RawAddition.compute(3, data(1.0, 2.0));
    });
  }

  @Test
  void testBackpropagation() {

    double[] input = data(111.1, 122.2, 113.3, 121.4, 112.5, 123.6);
    assertArrayEquals(input, RawAddition.gradient(input), DELTA);
  }

}
