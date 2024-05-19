package net.cristcost.differentiable;

import static net.cristcost.differentiable.MathLibrary.*;
import static net.cristcost.differentiable.TensorAsserts.assertTensorsEquals;
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

    VariableTensor x1 = scalar().variable().withData(2.0);
    ComputedTensor y1 = pow(x1, scalar().withData(8.0));
    y1.startBackpropagation();
    assertArrayEquals(data(8.0 * Math.pow(2.0, 7.0)), x1.getGradient());

    VariableTensor x2 = scalar().variable().withData(3.0);
    ComputedTensor y2 = pow(scalar().withData(Math.E), x2);
    y2.startBackpropagation();
    assertArrayEquals(data(Math.pow(Math.E, 3.0)), x2.getGradient());


    VariableTensor x3 = scalar().variable().withData(0.5);
    ComputedTensor y3 = pow(scalar().withData(4.0), x3);
    y3.startBackpropagation();
    assertArrayEquals(data(Math.log(4.0) * Math.pow(4.0, 0.5)), x3.getGradient());

    VariableTensor x41 = scalar().variable().withData(5.0);
    VariableTensor x42 = scalar().variable().withData(3.0);
    ComputedTensor y4 = pow(x41, x42);
    y4.startBackpropagation();
    assertArrayEquals(data(3.0 * Math.pow(5.0, 2.0)), x41.getGradient());
    assertArrayEquals(data(Math.log(5.0) * Math.pow(5.0, 3.0)), x42.getGradient());
    VariableTensor x5 = scalar().variable().withData(3.0);
    ComputedTensor y5 = pow(x5, x5);
    y5.startBackpropagation();
    assertArrayEquals(data((Math.log(3.0) + 1.0) * Math.pow(3.0, 3.0)), x5.getGradient());

  }

  @Test
  void testBasicReluOperation() {
    VariableTensor x1 = scalar().variable().withData(3.0);
    ComputedTensor y1 = relu(x1);
    y1.startBackpropagation();
    assertArrayEquals(data(1.0), x1.getGradient());

    VariableTensor x2 = scalar().variable().withData(-3.0);
    ComputedTensor y2 = relu(x2);
    y2.startBackpropagation();
    assertArrayEquals(data(0.0), x2.getGradient());
  }

  @Test
  void testDotProductOperation() {

    VariableTensor x1 = vector(1).variable().withData(3.0);
    ComputedTensor y1 = dot(x1, vector(2.0));
    assertTensorsEquals(scalar(6.0), y1);
    y1.startBackpropagation();
    assertArrayEquals(data(2.0), x1.getGradient());


    VariableTensor x2 = vector(3).variable().withData(3.0, 4.0, 5.0);
    ComputedTensor y2 = dot(x2, vector(2.0, 0.5, -1.0));
    assertTensorsEquals(scalar(3.0 * 2.0 + 0.5 * 4.0 - 5.0), y2);
    y2.startBackpropagation();
    assertArrayEquals(data(2.0, 0.5, -1.0), x2.getGradient());

    VariableTensor x3 = vector(3).variable().withData(3.0, 4.0, 5.0);
    ComputedTensor y3 = dot(x3, x3);
    assertTensorsEquals(scalar(9.0 + 16.0 + 25.0), y3);
    y3.startBackpropagation();
    assertArrayEquals(data(6.0, 8.0, 10.0), x3.getGradient());
  }

  @Test
  void testMseProductOperation() {
    VariableTensor x1 = vector(1).variable().withData(2.0);
    ComputedTensor y1 = mse(x1, vector(2.0));
    assertTensorsEquals(scalar(0.0), y1);
    y1.startBackpropagation();
    assertArrayEquals(data(0.0), x1.getGradient());

    VariableTensor x2 = vector(1).variable().withData(5.0);
    ComputedTensor y2 = mse(x2, vector(2.0));
    assertTensorsEquals(scalar(3.0 * 3.0), y2);
    y2.startBackpropagation();
    assertArrayEquals(data(6.0), x2.getGradient());

    VariableTensor x3 = vector(1).variable().withData(5.0);
    ComputedTensor y3 = mse(vector(2.0), x3);
    assertTensorsEquals(scalar(-3.0 * -3.0), y3);
    y3.startBackpropagation();
    assertArrayEquals(data(6.0), x3.getGradient());

    VariableTensor x4 = vector(3).variable().withData(3.0, 4.0, 5.0);
    ComputedTensor y4 = mse(x4, vector(2.0, 4.0, 5.5));
    assertTensorsEquals(scalar(0.4167), y4);
    y4.startBackpropagation();
    assertArrayEquals(data(0.6667, 0.0, -0.3333), x4.getGradient(), 0.0001);

    VariableTensor x51 = vector(3).variable().withData(3.0, 4.0, 5.0);
    VariableTensor x52 = vector(3).variable().withData(2.0, 4.0, 5.5);
    ComputedTensor y5 = mse(x51, x52);
    assertTensorsEquals(scalar(0.4167), y5);
    y5.startBackpropagation();
    assertArrayEquals(data(0.6667, 0.0, -0.3333), x51.getGradient(), 0.0001);
    assertArrayEquals(data(-0.6667, 0.0, 0.3333), x52.getGradient(), 0.0001);
  }

  @Test
  void testSoftMaxProductOperation() {

    VariableTensor x1 = vector(1).variable().withData(2.0);
    ComputedTensor y1 = softmax(x1);
    assertTensorsEquals(vector(1.0), y1);
    y1.startBackpropagation();
    assertArrayEquals(data(0.0), x1.getGradient());

    VariableTensor x2 = vector(3).variable().withData(5.0, 4.0, 3.0);
    ComputedTensor y2 = softmax(x2);
    assertTensorsEquals(vector(0.6652, 0.2447, 0.0900), y2);
    y2.backpropagate(data(1.0, 1.0, 1.0));
    assertArrayEquals(data(0.0, 0.0, 0.0), x2.getGradient());


    VariableTensor x3 = vector(3).variable().withData(0.0, Math.log(2.0), 0.0);
    ComputedTensor y3 = softmax(x3);
    assertTensorsEquals(vector(0.25, 0.5, 0.25), y3);
    y3.backpropagate(data(1.0, 1.0, 1.0));
    assertArrayEquals(data(0.0, 0.0, 0.0), x3.getGradient());

    VariableTensor x4 = vector(3).variable().withData(0.0, Math.log(2.0), 0.0);
    ComputedTensor y4 = softmax(x4);
    assertTensorsEquals(vector(0.25, 0.5, 0.25), y4);
    y4.backpropagate(data(1.0, 2.0, 1.0));
    assertArrayEquals(data(-0.125, 0.25, -0.125), x4.getGradient());

    VariableTensor x5 =
        vector(4).variable().withData(Math.log(4.0), Math.log(3.0), Math.log(2.0), Math.log(1.0));
    ComputedTensor y5 = softmax(x5);
    assertTensorsEquals(vector(0.4, 0.3, 0.2, 0.1), y5);
    y5.backpropagate(data(1.0, 1.0, 1.0, 1.0));
    assertArrayEquals(data(0.0, 0.0, 0.0, 0.0), x5.getGradient());

    VariableTensor x6 =
        vector(4).variable().withData(Math.log(4.0), Math.log(3.0), Math.log(2.0), Math.log(1.0));
    ComputedTensor y6 = softmax(x6);
    assertTensorsEquals(vector(0.4, 0.3, 0.2, 0.1), y6);
    y6.backpropagate(data(1.0, 1.0, 1.0, 10.0));
    assertArrayEquals(data(-0.3600, -0.2700, -0.1800, 0.8100), x6.getGradient());

  }

  @Test
  void testMatMulOperation() {
    fail("Test todo");
  }

  @Test
  void testComplexOperation() {

    // f(x) = relu(-0.5x^2 + 2x + 6)
    // df(x) = drelu(-0.5x^2 + 2x + 6) * d(-0.5x^2 + 2x + 6) = drelu(-0.5x^2 + 2x +6) * (-x + 2)

    Function<VariableTensor, VariableTensor> df = x -> {
      ComputedTensor y = relu(sum(
          multiply((scalar().withData(-0.5)),
              pow(x, scalar().withData(2.0))),
          multiply(scalar().withData(2.0), x),
          scalar().withData(6.0)));
      y.startBackpropagation();
      return x;
    };

    assertArrayEquals(data(0.0), df.apply(scalar().variable().withData(-4.0)).getGradient());
    assertArrayEquals(data(0.0), df.apply(scalar().variable().withData(-2.0)).getGradient());
    assertArrayEquals(data(2.0), df.apply(scalar().variable().withData(0.0)).getGradient());
    assertArrayEquals(data(0.0), df.apply(scalar().variable().withData(2.0)).getGradient());
    assertArrayEquals(data(-2.0), df.apply(scalar().variable().withData(4.0)).getGradient());
    assertArrayEquals(data(0.0), df.apply(scalar().variable().withData(6.0)).getGradient());
    assertArrayEquals(data(0.0), df.apply(scalar().variable().withData(8.0)).getGradient());

  }

}
