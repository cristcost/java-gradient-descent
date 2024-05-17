package net.cristcost.differentiable;

import static net.cristcost.differentiable.MathLibrary.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.function.Function;
import org.junit.jupiter.api.Test;

class MathLibraryTest {


  @Test
  void testSum() {
    assertTensorsEquals(scalar(3.0 + 5.0), sum(scalar(3.0), scalar(5.0)));
    assertTensorsEquals(scalar(7.0 - 12.0 + 18.0),
        sum(scalar(7.0), scalar(-12.0), scalar(18.0)));

  }

  @Test
  void testMultiply() {
    assertTensorsEquals(scalar(3.0 * 5.0), multiply(scalar(3.0), scalar(5.0)));
    assertTensorsEquals(scalar(0.5 * -0.3 * -1.0),
        multiply(scalar(0.5), scalar(-0.3), scalar(-1.0)));
  }

  @Test
  void testPow() {
    assertTensorsEquals(scalar(Math.pow(2.0, 8.0)), pow(scalar(2.0), scalar(8.0)));
    assertTensorsEquals(scalar(Math.pow(4.0, 0.5)), pow(scalar(4.0), scalar(0.5)));
    assertTensorsEquals(scalar(Math.pow(3.0, -1.0)), pow(scalar(3.0), scalar(-1.0)));
  }

  @Test
  void testRelu() {
    assertTensorsEquals(scalar(3.0), relu(scalar(3.0)));
    assertTensorsEquals(scalar(0.0), relu(scalar(-5.0)));
  }

  @Test
  void testCombinedOperation() {

    // f(x) = relu(0.5x^2 - 2x -6)

    Function<Tensor, Tensor> f = x -> relu(
        sum(
            multiply((scalar(-0.5)), pow(x, scalar(2.0))),
            multiply(scalar(2.0), x),
            scalar(6.0)));

    assertTensorsEquals(scalar(0.0), f.apply(scalar(-4.0)));
    assertTensorsEquals(scalar(0.0), f.apply(scalar(-2.0)));
    assertTensorsEquals(scalar(6.0), f.apply(scalar(0.0)));
    assertTensorsEquals(scalar(8.0), f.apply(scalar(2.0)));
    assertTensorsEquals(scalar(6.0), f.apply(scalar(4.0)));
    assertTensorsEquals(scalar(0.0), f.apply(scalar(6.0)));
    assertTensorsEquals(scalar(0.0), f.apply(scalar(8.0)));

  }


  public static void assertTensorsEquals(Tensor expected, Tensor actual) {
    assertArrayEquals(expected.getShape(), actual.getShape());
    assertArrayEquals(expected.getData(), actual.getData());
  }

}
