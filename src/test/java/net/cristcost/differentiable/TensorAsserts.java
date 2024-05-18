package net.cristcost.differentiable;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class TensorAsserts {

  public static void assertTensorsEquals(Tensor expected, Tensor actual) {
    assertArrayEquals(expected.getShape(), actual.getShape());
    assertArrayEquals(expected.getData(), actual.getData(), 0.00001);
  }

}
