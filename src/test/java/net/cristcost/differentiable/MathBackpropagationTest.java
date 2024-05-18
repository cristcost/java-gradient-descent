package net.cristcost.differentiable;

import static net.cristcost.differentiable.MathLibrary.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.function.Function;
import org.junit.jupiter.api.Test;

class MathLibraryBackpropagationTest {

  @Test
  void testSum() {
    VariableTensor x1 = scalar().variable().withData(3.0);
    ComputedTensor y1 = sum(x1, scalar().withData(5.0));
    y1.startBackpropagation();
    assertArrayEquals(data(1.0), x1.getGradient());

    VariableTensor x21 = scalar().variable().withData(7.0);
    VariableTensor x22 = scalar().variable().withData(-12.0);
    ComputedTensor y2 = sum(x21, x22, scalar().withData(18.0));
    y2.startBackpropagation();
    assertArrayEquals(data(1.0), x21.getGradient());
    assertArrayEquals(data(1.0), x22.getGradient());

    VariableTensor x3 = scalar().variable().withData(5.0);
    ComputedTensor y3 = sum(x3, x3, x3);
    y3.startBackpropagation();
    assertArrayEquals(data(3.0), x3.getGradient());


    VariableTensor x4 = matrix(2, 2).variable().withData(1.0, 2.0, 3.0, 4.0);
    ComputedTensor y4 = sum(x4, scalar().withData(5.0));
    y4.startBackpropagation();
    assertArrayEquals(data(1.0, 1.0, 1.0, 1.0), x4.getGradient());

    VariableTensor x5 = matrix(2, 2).variable().ones();
    ComputedTensor y5 = sum(x5, x5, matrix(2, 2).ones(), x5, scalar().withData(5.0),
        vector(2).withData(10.0, 20.0));
    y5.startBackpropagation();
    assertArrayEquals(data(3.0, 3.0, 3.0, 3.0), x5.getGradient());

  }

  @Test
  void testMultiply() {
    VariableTensor x1 = scalar().variable().withData(3.0);
    ComputedTensor y1 = multiply(x1, scalar().withData(5.0));
    y1.startBackpropagation();
    assertArrayEquals(data(5.0), x1.getGradient());

    VariableTensor x21 = scalar().variable().withData(0.5);
    VariableTensor x22 = scalar().variable().withData(-0.3);
    ComputedTensor y2 = multiply(x21, x22, scalar().withData(-1.0));
    y2.startBackpropagation();
    assertArrayEquals(data(0.3), x21.getGradient());
    assertArrayEquals(data(-0.5), x22.getGradient());

    VariableTensor x3 = scalar().variable().withData(2.0);
    ComputedTensor y3 = multiply(x3, x3, x3);
    y3.startBackpropagation();
    assertArrayEquals(data(2.0 * 2.0 * 3.0), x3.getGradient());


    VariableTensor x4 = matrix(2, 2).variable().withData(1.0, 2.0, 3.0, 4.0);
    ComputedTensor y4 = multiply(x4, matrix(2, 2).withData(5.0, 4.0, 3.0, 2.0));
    y4.startBackpropagation();
    assertArrayEquals(data(5.0, 4.0, 3.0, 2.0), x4.getGradient());

    VariableTensor x5 = matrix(2, 2).variable().ones();
    ComputedTensor y5 = multiply(x5, x5, matrix(2, 2).ones(), scalar().withData(2.0),
        vector(2).withData(-1.0, 0.5));
    y5.startBackpropagation();
    assertArrayEquals(data(-4.0, 2.0, -4.0, 2.0), x5.getGradient());
  }

  @Test
  void testBasicPowerOperation() {

    VariableTensor x1 = scalar().variable().withData(data(2.0));
    ComputedTensor y1 = pow(x1, scalar().withData(data(8.0)));
    y1.startBackpropagation();
    assertEquals(8.0 * Math.pow(2.0, 7.0), x1.getGradient());

    VariableTensor x2 = scalar().variable().withData(data(3.0));
    ComputedTensor y2 = pow(scalar().withData(data(Math.E)), x2);
    y2.startBackpropagation();
    assertEquals(Math.pow(Math.E, 3.0), x2.getGradient());


    VariableTensor x3 = scalar().variable().withData(data(0.5));
    ComputedTensor y3 = pow(scalar().withData(data(4.0)), x3);
    y3.startBackpropagation();
    assertEquals(Math.log(4.0) * Math.pow(4.0, 0.5), x3.getGradient());

    VariableTensor x41 = scalar().variable().withData(data(5.0));
    VariableTensor x42 = scalar().variable().withData(data(3.0));
    ComputedTensor y4 = pow(x41, x42);
    y4.startBackpropagation();
    assertEquals(3.0 * Math.pow(5.0, 2.0), x41.getGradient());
    assertEquals(Math.log(5.0) * Math.pow(5.0, 3.0), x42.getGradient());
    VariableTensor x5 = scalar().variable().withData(data(3.0));
    ComputedTensor y5 = pow(x5, x5);
    y5.startBackpropagation();
    assertEquals((Math.log(3.0) + 1.0) * Math.pow(3.0, 3.0), x5.getGradient());

  }

  @Test
  void testBasicReluOperation() {
    VariableTensor x1 = scalar().variable().withData(data(3.0));
    ComputedTensor y1 = relu(x1);
    y1.startBackpropagation();
    assertEquals(1.0, x1.getGradient());

    VariableTensor x2 = scalar().variable().withData(data(-3.0));
    ComputedTensor y2 = relu(x2);
    y2.startBackpropagation();
    assertEquals(0.0, x2.getGradient());
  }

  @Test
  void testComplexOperation() {

    // f(x) = relu(-0.5x^2 + 2x + 6)
    // df(x) = drelu(-0.5x^2 + 2x + 6) * d(-0.5x^2 + 2x + 6) = drelu(-0.5x^2 + 2x +6) * (-x + 2)

    Function<VariableTensor, VariableTensor> df = x -> {
      ComputedTensor y = relu(sum(
          multiply((scalar().withData(data(-0.5))),
              pow(x, scalar().withData(data(2.0)))),
          multiply(scalar().withData(data(2.0)), x),
          scalar().withData(data(6.0))));
      y.startBackpropagation();
      return x;
    };

    assertEquals(0.0, df.apply(scalar().variable().withData(data(-4.0))).getGradient());
    assertEquals(0.0, df.apply(scalar().variable().withData(data(-2.0))).getGradient());
    assertEquals(2.0, df.apply(scalar().variable().withData(data(0.0))).getGradient());
    assertEquals(0.0, df.apply(scalar().variable().withData(data(2.0))).getGradient());
    assertEquals(-2.0, df.apply(scalar().variable().withData(data(4.0))).getGradient());
    assertEquals(0.0, df.apply(scalar().variable().withData(data(6.0))).getGradient());
    assertEquals(0.0, df.apply(scalar().variable().withData(data(8.0))).getGradient());

  }

}
