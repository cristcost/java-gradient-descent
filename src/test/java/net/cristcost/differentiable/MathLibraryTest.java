package net.cristcost.differentiable;

import static net.cristcost.differentiable.MathLibrary.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import java.util.function.Function;
import org.junit.jupiter.api.Test;

class MathLibraryTest {


  @Test
  void testSum() {
    assertTensorsEquals(scalar().withData(3.0 + 5.0), sum(scalar().withData(3.0), scalar().withData(5.0)));
    assertTensorsEquals(scalar().withData(7.0 - 12.0 + 18.0),
        sum(scalar().withData(7.0), scalar().withData(-12.0), scalar().withData(18.0)));
    double[] value = {3.0 + 5.0};
    double[] value1 = {3.0};
    double[] value2 = {5.0};

    assertTensorsEquals(vector(value.length).withData(value), sum(vector(value1.length).withData(value1), vector(value2.length).withData(value2)));

    var matrix2by2 = matrix(2, 2);

    assertTensorsEquals(
        matrix2by2.withData(2.1, 4.2, 6.3, 8.4),
        sum(
            matrix2by2.withData(2.0, 4.0, 6.0, 8.0),
            matrix2by2.withData(0.1, 0.2, 0.3, 0.4)));

  }

  @Test
  void testMultiply() {
    assertTensorsEquals(scalar().withData(3.0 * 5.0), multiply(scalar().withData(3.0), scalar().withData(5.0)));
    assertTensorsEquals(scalar().withData(0.5 * -0.3 * -1.0),
        multiply(scalar().withData(0.5), scalar().withData(-0.3), scalar().withData(-1.0)));

    var matrix2by2 = matrix(2, 2);

    assertTensorsEquals(
        matrix2by2.withData(0.2, 0.8, 1.8, 3.2),
        multiply(
            matrix2by2.withData(2.0, 4.0, 6.0, 8.0),
            matrix2by2.withData(0.1, 0.2, 0.3, 0.4)));
  }

  @Test
  void testPow() {
    assertTensorsEquals(scalar().withData(Math.pow(2.0, 8.0)), pow(scalar().withData(2.0), scalar().withData(8.0)));
    assertTensorsEquals(scalar().withData(Math.pow(4.0, 0.5)), pow(scalar().withData(4.0), scalar().withData(0.5)));
    assertTensorsEquals(scalar().withData(Math.pow(3.0, -1.0)), pow(scalar().withData(3.0), scalar().withData(-1.0)));

    var matrix2by2 = matrix(2, 2);

    assertTensorsEquals(
        matrix2by2.withData(0.5, 2.0, 36.0, 1.0),
        pow(
            matrix2by2.withData(2.0, 4.0, 6.0, 8.0),
            matrix2by2.withData(-1.0, 0.5, 2.0, 0.0)));
  }

  @Test
  void testRelu() {
    assertTensorsEquals(scalar().withData(3.0), relu(scalar().withData(3.0)));
    assertTensorsEquals(scalar().withData(0.0), relu(scalar().withData(-5.0)));

    var matrix3by2 = matrix(3, 2);

    assertTensorsEquals(
        matrix3by2.withData(1.0, 0.0, 0.0, 0.0, 3.0, 6.0),
        relu(
            matrix3by2.withData(1.0, 0.0, -1.0, -2.0, 3.0, 6.0)));
  }


  @Test
  void testElementalBroadcasting() {
    double[] value = {2.0, 4.0};
    double[] value1 = {0.0, 2.0};
    assertTensorsEquals(
        vector(value.length).withData(value),
        sum(scalar().withData(2.0), vector(value1.length).withData(value1)));
    double[] value2 = {2.0, 4.0};
    double[] value3 = {0.0, 2.0};

    assertTensorsEquals(
        vector(value2.length).withData(value2),
        sum(vector(value3.length).withData(value3), scalar().withData(2.0)));
    double[] value4 = {2.0, 4.0};
    double[] value5 = {1.0, 2.0};

    assertTensorsEquals(
        vector(value4.length).withData(value4),
        multiply(scalar().withData(2.0), vector(value5.length).withData(value5)));
    double[] value6 = {2.0, 4.0};
    double[] value7 = {1.0, 2.0};

    assertTensorsEquals(
        vector(value6.length).withData(value6),
        multiply(vector(value7.length).withData(value7), scalar().withData(2.0)));
    double[] value8 = {4.0, 8.0};
    double[] value9 = {2.0, 3.0};

    assertTensorsEquals(
        vector(value8.length).withData(value8),
        pow(scalar().withData(2.0), vector(value9.length).withData(value9)));
    double[] value10 = {4.0, 9.0};
    double[] value11 = {2.0, 3.0};

    assertTensorsEquals(
        vector(value10.length).withData(value10),
        pow(vector(value11.length).withData(value11), scalar().withData(2.0)));
  }

  @Test
  void testCombinedOperation() {

    // f(x) = relu(0.5x^2 - 2x -6)

    Function<Tensor, Tensor> f = x -> relu(
        sum(
            multiply((scalar().withData(-0.5)), pow(x, scalar().withData(2.0))),
            multiply(scalar().withData(2.0), x),
            scalar().withData(6.0)));

    assertTensorsEquals(scalar().withData(0.0), f.apply(scalar().withData(-4.0)));
    assertTensorsEquals(scalar().withData(0.0), f.apply(scalar().withData(-2.0)));
    assertTensorsEquals(scalar().withData(6.0), f.apply(scalar().withData(0.0)));
    assertTensorsEquals(scalar().withData(8.0), f.apply(scalar().withData(2.0)));
    assertTensorsEquals(scalar().withData(6.0), f.apply(scalar().withData(4.0)));
    assertTensorsEquals(scalar().withData(0.0), f.apply(scalar().withData(6.0)));
    assertTensorsEquals(scalar().withData(0.0), f.apply(scalar().withData(8.0)));
    double[] value = {0.0, 0.0, 6.0, 8.0, 6.0, 0.0, 0.0};
    double[] value1 = {-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0};


    assertTensorsEquals(
        vector(value.length).withData(value),
        f.apply(vector(value1.length).withData(value1)));

  }


  public static void assertTensorsEquals(Tensor expected, Tensor actual) {
    assertArrayEquals(expected.getShape(), actual.getShape());
    assertArrayEquals(expected.getData(), actual.getData(), 0.00001);
  }

}
