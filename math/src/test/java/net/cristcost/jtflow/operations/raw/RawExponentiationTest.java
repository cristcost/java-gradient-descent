package net.cristcost.jtflow.operations.raw;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class RawExponentiationTest {


  private static final double DELTA = 1e-5;


  private static double[] data(double... data) {
    return data;
  }

  @Test
  void testOperation() {

    assertArrayEquals(data(1.0, 4.0, 9.0),
        RawExponentiation.compute(data(1.0, 2.0, 3.0), data(2.0)),
        DELTA);

    assertArrayEquals(data(1.0, 1.0, 4.0, Math.sqrt(2.0), 9.0, Math.sqrt(3.0)),
        RawExponentiation.compute(data(1.0, 1.0, 2.0, 2.0, 3.0, 3.0), data(2.0, 0.5)),
        DELTA);


    assertThrows(RuntimeException.class, () -> {
      RawExponentiation.compute(data(1.0), data(1.0, 2.0));
    });
  }

  @Test
  void testBackpropagationBase() {

    assertArrayEquals(data(0.0),
        RawExponentiation.baseGradient(data(1.0), data(0.0), data(2.0)), DELTA);
    assertArrayEquals(data(2.0),
        RawExponentiation.baseGradient(data(1.0), data(1.0), data(2.0)), DELTA);
    assertArrayEquals(data(4.0),
        RawExponentiation.baseGradient(data(1.0), data(2.0), data(2.0)), DELTA);

    assertArrayEquals(data(0.0, 2.0, 4.0),
        RawExponentiation.baseGradient(data(1.0, 1.0, 1.0), data(0.0, 1.0, 2.0), data(2.0)), DELTA);

    assertArrayEquals(data(0.0, 4.0, 8.0),
        RawExponentiation.baseGradient(data(2.0, 2.0, 2.0), data(0.0, 1.0, 2.0), data(2.0)), DELTA);

  }

  @Test
  void testBackpropagationExponent() {

    assertArrayEquals(data(0.0),
        RawExponentiation.exponentGradient(data(1.0), data(1.0), data(2.0)), DELTA);
    assertArrayEquals(data(Math.pow(2.0, 2.0) * Math.log(2.0)),
        RawExponentiation.exponentGradient(data(1.0), data(2.0), data(2.0)), DELTA);


    assertArrayEquals(data(
        Math.pow(1.0, 2.0) * Math.log(1.0),
        Math.pow(2.0, 2.0) * Math.log(2.0),
        Math.pow(3.0, 2.0) * Math.log(3.0)),
        RawExponentiation.exponentGradient(data(1.0, 1.0, 1.0), data(1.0, 2.0, 3.0), data(2.0)),
        DELTA);

    assertArrayEquals(data(
        Math.pow(2.0, 1.0) * Math.log(2.0),
        Math.pow(2.0, 2.0) * Math.log(2.0),
        Math.pow(2.0, 3.0) * Math.log(2.0)),
        RawExponentiation.exponentGradient(data(1.0, 1.0, 1.0), data(2.0, 2.0, 2.0),
            data(1.0, 2.0, 3.0)),
        DELTA);



  }
}
