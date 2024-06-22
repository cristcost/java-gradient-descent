package net.cristcost.jtflow.operations.raw;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class RawMultiplicationTest {

  private static final double DELTA = 1e-5;


  private static double[] data(double... data) {
    return data;
  }

  @Test
  void testOperation() {

    assertArrayEquals(data(1.0, 1.0, 1.0, 1.0, 1.0), RawMultiplication.compute(5, data(1.0)),
        DELTA);

    assertArrayEquals(data(5.0, 5.0, 5.0, 5.0, 5.0),
        RawMultiplication.compute(5, data(5.0), data(1.0), data(1.0), data(1.0), data(1.0)), DELTA);

    assertArrayEquals(data(100.0, 800.0, 900.0, 800.0, 1000.0, 3600.0),
        RawMultiplication.compute(6,
            data(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            data(1.0, 2.0, 3.0),
            data(10.0, 20.0),
            data(100.0)),
        DELTA);

    assertThrows(RuntimeException.class, () -> {
      RawMultiplication.compute(3, data(1.0, 2.0));
    });
  }

  @Test
  void testBackpropagationBase() {

    double[] input = data(1.0, 1.0);
    double[][] operands = {data(1.0, 1.0), data(1.0, 1.0)};

    assertArrayEquals(data(1.0, 1.0), RawMultiplication.gradient(input, 0, operands), DELTA);
    assertArrayEquals(data(1.0, 1.0), RawMultiplication.gradient(input, 1, operands), DELTA);

  }

  @Test
  void testBackpropagation() {

    double[] input = data(1.0, 1.0);
    double[][] operands = {data(2.0, 4.0), data(0.5, 0.25)};

    assertArrayEquals(data(0.5, 0.25), RawMultiplication.gradient(input, 0, operands), DELTA);
    assertArrayEquals(data(2.0, 4.0), RawMultiplication.gradient(input, 1, operands), DELTA);

  }

  @Test
  void testBackpropagationChainedGradient() {

    double[] input = data(2.0, 4.0);
    double[][] operands = {data(2.0, 4.0), data(0.5, 0.25)};

    assertArrayEquals(data(1.0, 1.0), RawMultiplication.gradient(input, 0, operands), DELTA);
    assertArrayEquals(data(4.0, 16.0), RawMultiplication.gradient(input, 1, operands), DELTA);

  }

  @Test
  void testBackpropagationChainedComplex() {

    double[] input = data(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    double[][] operands = {
        data(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        data(1.0, 2.0, 3.0),
        data(10.0, 20.0),
        data(100.0)};


    assertArrayEquals(data(1000.0, 8000.0, 9000.0, 8000.0, 10000.0, 36000.0),
        RawMultiplication.gradient(input, 0, operands), DELTA);
    assertArrayEquals(data(100.0, 800.0, 900.0, 3200.0, 2500.0, 7200.0),
        RawMultiplication.gradient(input, 1, operands), DELTA);
    assertArrayEquals(data(10.0, 80.0, 270.0, 160.0, 500.0, 1080.0),
        RawMultiplication.gradient(input, 2, operands), DELTA);
    assertArrayEquals(data(1.0, 16.0, 27.0, 32.0, 50.0, 216.0),
        RawMultiplication.gradient(input, 3, operands), DELTA);

  }
}
