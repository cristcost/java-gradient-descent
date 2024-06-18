package net.cristcost.jtflow;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import net.cristcost.jtflow.api.Tensor;

public class TensorAsserts {

  public static void assertTensorsEquals(Tensor expected, Tensor actual) {
    assertTensorsEquals(expected, actual, 0.0001);
  }

  private static void assertTensorsEquals(Tensor expected, Tensor actual, double delta) {
    assertArrayEquals(expected.getShape(), actual.getShape());
    assertArrayEquals(expected.getData(), actual.getData(), delta);
  }

}
