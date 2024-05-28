package net.cristcost.jtflow.operations.raw;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import org.junit.jupiter.api.Test;

class RawSoftMaxTest {

  private static final double DELTA = 1e-5;

  private static double[] data(double... data) {
    return data;
  }

  @Test
  void testBackpropagation() {

    assertArrayEquals(data(-6.867849e-08, -2.5265404e-08, -2.5265404e-08),
        RawSoftMax.gradient(data(1.0, 1.0, 1.0), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(0.0, 0.0, 0.0),
        RawSoftMax.gradient(data(1.0, 1.0, 1.0), data(1.0, 1.0, 1.0)), DELTA);
    assertArrayEquals(data(0.0, 0.0, 0.0),
        RawSoftMax.gradient(data(1.0, 1.0, 1.0), data(1.0, -2.0, 3.0)), DELTA);
    assertArrayEquals(data(0.0, 0.0, 0.0),
        RawSoftMax.gradient(data(1.0, 1.0, 1.0), data(1.0, 10.0, 100.0)), DELTA);
    assertArrayEquals(data(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        RawSoftMax.gradient(data(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), data(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
        DELTA);
    assertArrayEquals(data(0.36630908, -0.07718399, -0.28912556),
        RawSoftMax.gradient(data(5.0, 4.0, 3.0), data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(0.33333334, 0.0, -0.33333334),
        RawSoftMax.gradient(data(5.0, 4.0, 3.0), data(1.0, 1.0, 1.0)), DELTA);
    assertArrayEquals(data(0.20821586, 0.004466707, -0.21268246),
        RawSoftMax.gradient(data(5.0, 4.0, 3.0), data(1.0, -2.0, 3.0)), DELTA);
    assertArrayEquals(data(2.02e-43, 8.19401e-40, 0.0),
        RawSoftMax.gradient(data(5.0, 4.0, 3.0), data(1.0, 10.0, 100.0)), DELTA);
    assertArrayEquals(
        data(-0.0101551255, -0.039210957, -0.13813606, -0.118210375, -0.0882071, 0.39391953),
        RawSoftMax.gradient(data(5.0, 4.0, 3.0, 6.0, 7.0, 8.0), data(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
        DELTA);

  }

  @Test
  void testOperation() {

    assertArrayEquals(data(0.5761169, 0.21194157, 0.21194157),
        RawSoftMax.compute(data(1.0, 0.0, 0.0)), DELTA);
    assertArrayEquals(data(0.33333334, 0.33333334, 0.33333334),
        RawSoftMax.compute(data(1.0, 1.0, 1.0)), DELTA);
    assertArrayEquals(data(0.11849965, 0.00589975, 0.8756006),
        RawSoftMax.compute(data(1.0, -2.0, 3.0)), DELTA);
    assertArrayEquals(data(1.01e-43, 8.19401e-40, 1.0), RawSoftMax.compute(data(1.0, 10.0, 100.0)),
        DELTA);
    assertArrayEquals(
        data(0.0042697783, 0.011606461, 0.031549633, 0.085760795, 0.233122, 0.6336913),
        RawSoftMax.compute(data(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)), DELTA);
  }
}
