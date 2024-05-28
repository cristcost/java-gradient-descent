package net.cristcost.jtflow.operations.raw;

import static net.cristcost.jtflow.operations.raw.RawCategoricalCrossentropy.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class RawCategoricalCrossentropyTest {

  private static final double DELTA = 1e-5;

  private static double[] data(double... data) {
    return data;
  }

  @Test
  void testBackpropagation() {

    assertArrayEquals(data(-0.42388308, 0.21194157, 0.21194157),
        gradient(1.0, data(1.0, 0.0, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(0.5761169, -0.7880584, 0.21194157),
        gradient(1.0, data(1.0, 0.0, 0.0), data(0.0, 1.0, 0.0)), DELTA);
    assertArrayEquals(data(-0.46117747, 0.24210858, 0.2190689),
        gradient(1.0, data(0.9, 0.1, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(-0.6666667, 0.3333333, 0.3333333),
        gradient(1.0, data(0.3333, 0.3333, 0.3333), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(-0.64408696, 0.32204345, 0.32204345),
        gradient(1.0, data(0.4, 0.3, 0.3), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(-0.5727664, 0.2863832, 0.2863832),
        gradient(1.0, data(0.6000001, 0.20000002, 0.20000002), data(1.0, 0.0, 0.0)),
        DELTA);

    assertArrayEquals(data(-2.1194153, 1.0597079, 1.0597079),
        gradient(5.0, data(1.0, 0.0, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(2.8805847, -3.9402921, 1.0597079),
        gradient(5.0, data(1.0, 0.0, 0.0), data(0.0, 1.0, 0.0)), DELTA);
    assertArrayEquals(data(-2.3058872, 1.2105429, 1.0953445),
        gradient(5.0, data(0.9, 0.1, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(-3.3333335, 1.6666665, 1.6666665),
        gradient(5.0, data(0.3333, 0.3333, 0.3333), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(-3.2204347, 1.6102172, 1.6102172),
        gradient(5.0, data(0.4, 0.3, 0.3), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(-2.863832, 1.431916, 1.431916),
        gradient(5.0, data(0.6000001, 0.20000002, 0.20000002), data(1.0, 0.0, 0.0)),
        DELTA);


  }

  @Test
  void testOperation() {

    assertEquals(0.5514446496963501, compute(data(1.0, 0.0, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertEquals(1.55144464969635, compute(data(1.0, 0.0, 0.0), data(0.0, 1.0, 0.0)), DELTA);
    assertEquals(0.6183689832687378, compute(data(0.9, 0.1, 0.0), data(1.0, 0.0, 0.0)), DELTA);
    assertEquals(1.0986123085021973, compute(data(0.3333, 0.3333, 0.3333), data(1.0, 0.0, 0.0)),
        DELTA);
    assertEquals(1.0330687761306763, compute(data(0.4, 0.3, 0.3), data(1.0, 0.0, 0.0)), DELTA);
    assertEquals(0.8504244089126587,
        compute(data(0.6000001, 0.20000002, 0.20000002), data(1.0, 0.0, 0.0)), DELTA);
  }

}
