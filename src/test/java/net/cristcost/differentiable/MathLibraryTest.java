package net.cristcost.differentiable;

import static net.cristcost.differentiable.MathLibrary.*;
import java.util.function.Function;
import org.junit.jupiter.api.Test;

class MathLibraryTest {


  @Test
  void testSum() {
    TensorAsserts.assertTensorsEquals(scalar().withData(3.0 + 5.0),
        sum(scalar().withData(3.0), scalar().withData(5.0)));
    TensorAsserts.assertTensorsEquals(scalar().withData(7.0 - 12.0 + 18.0),
        sum(scalar().withData(7.0), scalar().withData(-12.0), scalar().withData(18.0)));
    TensorAsserts.assertTensorsEquals(vector(1).withData(3.0 + 5.0),
        sum(vector(1).withData(3.0), vector(1).withData(5.0)));

    var matrix2by2 = matrix(2, 2);

    TensorAsserts.assertTensorsEquals(
        matrix2by2.withData(2.1, 4.2, 6.3, 8.4),
        sum(
            matrix2by2.withData(2.0, 4.0, 6.0, 8.0),
            matrix2by2.withData(0.1, 0.2, 0.3, 0.4)));

  }

  @Test
  void testMultiply() {
    TensorAsserts.assertTensorsEquals(scalar().withData(3.0 * 5.0),
        multiply(scalar().withData(3.0), scalar().withData(5.0)));
    TensorAsserts.assertTensorsEquals(scalar().withData(0.5 * -0.3 * -1.0),
        multiply(scalar().withData(0.5), scalar().withData(-0.3), scalar().withData(-1.0)));

    var matrix2by2 = matrix(2, 2);

    TensorAsserts.assertTensorsEquals(
        matrix2by2.withData(0.2, 0.8, 1.8, 3.2),
        multiply(
            matrix2by2.withData(2.0, 4.0, 6.0, 8.0),
            matrix2by2.withData(0.1, 0.2, 0.3, 0.4)));
  }

  @Test
  void testPow() {
    TensorAsserts.assertTensorsEquals(scalar().withData(Math.pow(2.0, 8.0)),
        pow(scalar().withData(2.0), scalar().withData(8.0)));
    TensorAsserts.assertTensorsEquals(scalar().withData(Math.pow(4.0, 0.5)),
        pow(scalar().withData(4.0), scalar().withData(0.5)));
    TensorAsserts.assertTensorsEquals(scalar().withData(Math.pow(3.0, -1.0)),
        pow(scalar().withData(3.0), scalar().withData(-1.0)));

    var matrix2by2 = matrix(2, 2);

    TensorAsserts.assertTensorsEquals(
        matrix2by2.withData(0.5, 2.0, 36.0, 1.0),
        pow(
            matrix2by2.withData(2.0, 4.0, 6.0, 8.0),
            matrix2by2.withData(-1.0, 0.5, 2.0, 0.0)));
  }

  @Test
  void testRelu() {
    TensorAsserts.assertTensorsEquals(scalar().withData(3.0), relu(scalar().withData(3.0)));
    TensorAsserts.assertTensorsEquals(scalar().withData(0.0), relu(scalar().withData(-5.0)));

    var matrix3by2 = matrix(3, 2);

    TensorAsserts.assertTensorsEquals(
        matrix3by2.withData(1.0, 0.0, 0.0, 0.0, 3.0, 6.0),
        relu(
            matrix3by2.withData(1.0, 0.0, -1.0, -2.0, 3.0, 6.0)));
  }


  @Test
  void testElementalBroadcasting() {
    TensorAsserts.assertTensorsEquals(
        vector(2).withData(2.0, 4.0),
        sum(scalar().withData(2.0), vector(2).withData(0.0, 2.0)));
    TensorAsserts.assertTensorsEquals(
        vector(2).withData(2.0, 4.0),
        sum(vector(2).withData(0.0, 2.0), scalar().withData(2.0)));
    TensorAsserts.assertTensorsEquals(
        vector(2).withData(2.0, 4.0),
        multiply(scalar().withData(2.0), vector(2).withData(1.0, 2.0)));
    TensorAsserts.assertTensorsEquals(
        vector(2).withData(2.0, 4.0),
        multiply(vector(2).withData(1.0, 2.0), scalar().withData(2.0)));
    TensorAsserts.assertTensorsEquals(
        vector(2).withData(4.0, 8.0),
        pow(scalar().withData(2.0), vector(2).withData(2.0, 3.0)));
    TensorAsserts.assertTensorsEquals(
        vector(2).withData(4.0, 9.0),
        pow(vector(2).withData(2.0, 3.0), scalar().withData(2.0)));
  }

  @Test
  void testCombinedOperation() {

    // f(x) = relu(0.5x^2 - 2x -6)

    Function<Tensor, Tensor> f = x -> relu(
        sum(
            multiply((scalar().withData(-0.5)), pow(x, scalar().withData(2.0))),
            multiply(scalar().withData(2.0), x),
            scalar().withData(6.0)));

    TensorAsserts.assertTensorsEquals(scalar().withData(0.0), f.apply(scalar().withData(-4.0)));
    TensorAsserts.assertTensorsEquals(scalar().withData(0.0), f.apply(scalar().withData(-2.0)));
    TensorAsserts.assertTensorsEquals(scalar().withData(6.0), f.apply(scalar().withData(0.0)));
    TensorAsserts.assertTensorsEquals(scalar().withData(8.0), f.apply(scalar().withData(2.0)));
    TensorAsserts.assertTensorsEquals(scalar().withData(6.0), f.apply(scalar().withData(4.0)));
    TensorAsserts.assertTensorsEquals(scalar().withData(0.0), f.apply(scalar().withData(6.0)));
    TensorAsserts.assertTensorsEquals(scalar().withData(0.0), f.apply(scalar().withData(8.0)));
    TensorAsserts.assertTensorsEquals(
        vector(7).withData(0.0, 0.0, 6.0, 8.0, 6.0, 0.0, 0.0),
        f.apply(vector(7).withData(-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0)));

  }

}
