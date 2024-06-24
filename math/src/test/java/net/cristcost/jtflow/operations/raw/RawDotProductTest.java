package net.cristcost.jtflow.operations.raw;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class RawDotProductTest {

  private static final double DELTA = 1e-5;


  private static double[] data(double... data) {
    return data;
  }

  @Test
  void testOperation() {

    assertEquals(1.0, RawDotProduct.compute(data(1.0), data(1.0)), DELTA);

    assertEquals(
        1.0 * 1.0 - 2.0 * 2.0 + 3.0 * 0.5,
        RawDotProduct.compute(data(1.0, 2.0, 3.0), data(1.0, -2.0, 0.5)), DELTA);


    assertEquals(
        1.0 * 1.0 - 2.0 * 2.0 + 1.0 * 0.5 + 2.0 * 4.0,
        RawDotProduct.compute(data(1.0, 2.0), data(1.0, -2.0, 0.5, 4.0)), DELTA);

    assertThrows(RuntimeException.class, () -> {
      RawDotProduct.compute(data(1.0, 2.0), data(1.0, 2.0, 3.0));
    });
  }

  @Test
  void testBackpropagation() {
    assertArrayEquals(data(1.0), RawDotProduct.gradient(1.0, data(1.0)), DELTA);
    assertArrayEquals(data(1.0, 2.0, 3.0), RawDotProduct.gradient(1.0, data(1.0, 2.0, 3.0)), DELTA);
    assertArrayEquals(data(2.0, 4.0, 6.0), RawDotProduct.gradient(2.0, data(1.0, 2.0, 3.0)), DELTA);
  }

  @Test
  void test() {

    double[] x11 = {3.0};
    double[] x12 = {2.0};

    assertEquals(6.0, RawDotProduct.compute(x11, x12));
    assertArrayEquals(data(2.0), RawDotProduct.gradient(1.0, x12));
    assertArrayEquals(data(3.0), RawDotProduct.gradient(1.0, x11));


    double[] x21 = {3.0, 4.0, 5.0};
    double[] x22 = {2.0, 0.5, -1.0};

    assertEquals(3.0 * 2.0 + 0.5 * 4.0 - 5.0, RawDotProduct.compute(x21, x22));
    assertArrayEquals(data(2.0, 0.5, -1.0), RawDotProduct.gradient(1.0, x22));
    assertArrayEquals(data(3.0, 4.0, 5.0), RawDotProduct.gradient(1.0, x21));
  }
}
