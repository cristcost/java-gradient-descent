package net.cristcost.differentiable;

import static net.cristcost.differentiable.MathLibrary.*;
import static net.cristcost.differentiable.TensorAsserts.assertTensorsEquals;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Arrays;
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
  void testSoftMaxOperation() {

    VariableTensor x1 = vector(2).variable().withData(Math.log(3.0), Math.log(1.0));
    ComputedTensor y1 = softmax(x1);
    y1.backpropagate(data(1.0, 1.0));
    assertTensorsEquals(vector(0.75, 0.25), y1);
    assertArrayEquals(data(0.0, 0.0), x1.getGradient(), 0.0001);

    VariableTensor x2 = vector(2).variable().withData(Math.log(3.0), Math.log(1.0));
    ComputedTensor y2 = softmax(x2);
    y2.backpropagate(data(4.0, 0.0));
    assertTensorsEquals(vector(0.75, 0.25), y2);
    assertArrayEquals(data(0.75, -0.75), x2.getGradient(), 0.0001);


    VariableTensor x3 = vector(1).variable().withData(2.0);
    ComputedTensor y3 = softmax(x3);
    assertTensorsEquals(vector(1.0), y3);
    y3.startBackpropagation();
    assertArrayEquals(data(0.0), x3.getGradient(), 0.0001);

    VariableTensor x4 = vector(3).variable().withData(5.0, 4.0, 3.0);
    ComputedTensor y4 = softmax(x4);
    assertTensorsEquals(vector(0.6652, 0.2447, 0.0900), y4);
    y4.backpropagate(data(1.0, 1.0, 1.0));
    assertArrayEquals(data(0.0, 0.0, 0.0), x4.getGradient(), 0.0001);


    VariableTensor x5 = vector(3).variable().withData(0.0, Math.log(2.0), 0.0);
    ComputedTensor y5 = softmax(x5);
    assertTensorsEquals(vector(0.25, 0.5, 0.25), y5);
    y5.backpropagate(data(1.0, 1.0, 1.0));
    assertArrayEquals(data(0.0, 0.0, 0.0), x5.getGradient(), 0.0001);

    VariableTensor x6 = vector(3).variable().withData(0.0, Math.log(2.0), 0.0);
    ComputedTensor y6 = softmax(x6);
    assertTensorsEquals(vector(0.25, 0.5, 0.25), y6);
    y6.backpropagate(data(1.0, 2.0, 1.0));
    assertArrayEquals(data(-0.125, 0.25, -0.125), x6.getGradient(), 0.0001);

    VariableTensor x7 =
        vector(4).variable().withData(Math.log(4.0), Math.log(3.0), Math.log(2.0), Math.log(1.0));
    ComputedTensor y7 = softmax(x7);
    assertTensorsEquals(vector(0.4, 0.3, 0.2, 0.1), y7);
    y7.backpropagate(data(1.0, 1.0, 1.0, 1.0));
    assertArrayEquals(data(0.0, 0.0, 0.0, 0.0), x7.getGradient(), 0.0001);

    VariableTensor x8 =
        vector(4).variable().withData(Math.log(4.0), Math.log(3.0), Math.log(2.0), Math.log(1.0));
    ComputedTensor y8 = softmax(x8);
    assertTensorsEquals(vector(0.4, 0.3, 0.2, 0.1), y8);
    y8.backpropagate(data(1.0, 1.0, 1.0, 10.0));
    assertArrayEquals(data(-0.3600, -0.2700, -0.1800, 0.8100), x8.getGradient(), 0.0001);

  }

  @Test
  void testMatMulOperation() {


    VariableTensor x11 = matrix(2, 3).variable().withData(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    VariableTensor x12 = matrix(3, 2).variable().withData(2.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    ComputedTensor y1 = matmul(x11, x12);
    assertTensorsEquals(matrix(2, 2).withData(2.0, 0.0, 0.0, 0.0), y1);
    y1.backpropagate(matrix(2, 2).ones().getData());
    assertArrayEquals(data(2.0, 0.0, 0.0, 2.0, 0.0, 0.0), x11.getGradient());
    assertArrayEquals(data(1.0, 1.0, 0.0, 0.0, 0.0, 0.0), x12.getGradient());

    VariableTensor x21 = matrix(2, 2).variable().withData(1.0, 0.0, 1.0, 0.0);
    VariableTensor x22 = matrix(2, 2).variable().withData(0.0, 0.0, 0.0, 0.0);
    ComputedTensor y2 = matmul(x21, x22);
    assertTensorsEquals(matrix(2, 2).withData(0.0, 0.0, 0.0, 0.0), y2);
    y2.backpropagate(matrix(2, 2).ones().getData());
    assertArrayEquals(data(0.0, 0.0, 0.0, 0.0), x21.getGradient());
    assertArrayEquals(data(2.0, 2.0, 0.0, 0.0), x22.getGradient());

    VariableTensor x31 = matrix(2, 2).variable().withData(1.0, 2.0, 3.0, 4.0);
    VariableTensor x32 = matrix(2, 2).variable().withData(0.0, 0.0, 0.0, 0.0);
    ComputedTensor y3 = matmul(x31, x32);
    assertTensorsEquals(matrix(2, 2).withData(0.0, 0.0, 0.0, 0.0), y3);
    y3.backpropagate(matrix(2, 2).ones().getData());
    assertArrayEquals(data(0.0, 0.0, 0.0, 0.0), x31.getGradient());
    assertArrayEquals(data(4.0, 4.0, 6.0, 6.0), x32.getGradient());

    VariableTensor x41 = matrix(2, 2).variable().withData(0.0, 0.0, 0.0, 0.0);
    VariableTensor x42 = matrix(2, 2).variable().withData(1.0, 2.0, 3.0, 4.0);
    ComputedTensor y4 = matmul(x41, x42);
    assertTensorsEquals(matrix(2, 2).withData(0.0, 0.0, 0.0, 0.0), y4);
    y4.backpropagate(matrix(2, 2).ones().getData());
    assertArrayEquals(data(3.0, 7.0, 3.0, 7.0), x41.getGradient());
    assertArrayEquals(data(0.0, 0.0, 0.0, 0.0), x42.getGradient());

    VariableTensor x51 = matrix(2, 2).variable().withData(2.0, 4.0, 6.0, 8.0);
    VariableTensor x52 = matrix(2, 2).variable().withData(1.0, 3.0, 5.0, 7.0);
    ComputedTensor y5 = matmul(x51, x52);
    assertTensorsEquals(matrix(2, 2).withData(22.0, 34.0, 46.0, 74.0), y5);
    y5.backpropagate(matrix(2, 2).ones().getData());
    assertArrayEquals(data(4.0, 12.0, 4.0, 12.0), x51.getGradient());
    assertArrayEquals(data(8.0, 8.0, 12.0, 12.0), x52.getGradient());

    // Tests below generated in pytorch with
    // ```
    // import torch
    //
    // def data(tensor):
    //   return ", ".join(map(str, [x for x in tensor.flatten().detach().numpy()]))
    //
    //
    // m,n,p = torch.randint(low=1, high=6, size=(3,))
    // x1 = torch.randint(low=-10, high=11,size =(m,n), dtype=torch.float, requires_grad=True)
    // x2 = torch.randint(low=-10, high=11,size =(n,p), dtype=torch.float, requires_grad=True)
    // y = torch.matmul(x1, x2)
    // y.retain_grad()
    // z = y.sum()
    // z.backward()
    //
    // i = i + 1
    //
    // print(f"VariableTensor x{i}1 = matrix({m}, {n}).variable().withData({data(x1)});")
    // print(f"VariableTensor x{i}2 = matrix({n}, {p}).variable().withData({data(x2)});")
    // print(f"ComputedTensor y{i} = matmul(x{i}1, x{i}2);")
    // print(f"assertTensorsEquals(matrix({m}, {p}).withData({data(y)}), y{i});")
    // print(f"y{i}.backpropagate(matrix({m}, {p}).ones().getData());")
    // print(f"System.out.println(Arrays.toString(x{i}1.getGradient()));")
    // print(f"System.out.println(Arrays.toString(x{i}2.getGradient()));")
    // print(f"assertArrayEquals(data({data(x1.grad)}), x{i}1.getGradient());")
    // print(f"assertArrayEquals(data({data(x2.grad)}), x{i}2.getGradient());")
    // ```
    VariableTensor x61 = matrix(3, 2).variable().withData(2.0, 4.0, 6.0, 8.0, 10.0, 12.0);
    VariableTensor x62 = matrix(2, 3).variable().withData(1.0, 3.0, 5.0, 7.0, 9.0, 11.0);
    ComputedTensor y6 = matmul(x61, x62);
    assertTensorsEquals(
        matrix(3, 3).withData(30.0, 42.0, 54.0, 62.0, 90.0, 118.0, 94.0, 138.0, 182.0), y6);
    y6.backpropagate(matrix(3, 3).ones().getData());
    assertArrayEquals(data(9.0, 27.0, 9.0, 27.0, 9.0, 27.0), x61.getGradient());
    assertArrayEquals(data(18.0, 18.0, 18.0, 24.0, 24.0, 24.0), x62.getGradient());

    VariableTensor x71 = matrix(3, 4).variable().withData(-3.0, 3.0, 9.0, -4.0, -5.0, -8.0, 2.0,
        -10.0, 10.0, -6.0, -3.0, -4.0);
    VariableTensor x72 = matrix(4, 3).variable().withData(3.0, -7.0, -3.0, -10.0, -10.0, 5.0, 5.0,
        10.0, -3.0, 5.0, -5.0, 4.0);
    ComputedTensor y7 = matmul(x71, x72);
    assertTensorsEquals(
        matrix(3, 3).withData(-14.0, 101.0, -19.0, 25.0, 185.0, -71.0, 55.0, -20.0, -67.0), y7);
    y7.backpropagate(matrix(3, 3).ones().getData());
    System.out.println(Arrays.toString(x71.getGradient()));
    System.out.println(Arrays.toString(x72.getGradient()));
    assertArrayEquals(data(-7.0, -15.0, 12.0, 4.0, -7.0, -15.0, 12.0, 4.0, -7.0, -15.0, 12.0, 4.0),
        x71.getGradient());
    assertArrayEquals(data(2.0, 2.0, 2.0, -11.0, -11.0, -11.0, 8.0, 8.0, 8.0, -18.0, -18.0, -18.0),
        x72.getGradient());

    VariableTensor x81 = matrix(5, 3).variable().withData(-2.0, 3.0, 10.0, 4.0, 2.0, 2.0, 10.0,
        -1.0, -1.0, 2.0, -10.0, -5.0, 6.0, 7.0, 6.0);
    VariableTensor x82 = matrix(3, 2).variable().withData(-6.0, 8.0, 10.0, -6.0, 6.0, 9.0);
    ComputedTensor y8 = matmul(x81, x82);
    assertTensorsEquals(
        matrix(5, 2).withData(102.0, 56.0, 8.0, 38.0, -76.0, 77.0, -142.0, 31.0, 70.0, 60.0), y8);
    y8.backpropagate(matrix(5, 2).ones().getData());
    System.out.println(Arrays.toString(x81.getGradient()));
    System.out.println(Arrays.toString(x82.getGradient()));
    assertArrayEquals(
        data(2.0, 4.0, 15.0, 2.0, 4.0, 15.0, 2.0, 4.0, 15.0, 2.0, 4.0, 15.0, 2.0, 4.0, 15.0),
        x81.getGradient());
    assertArrayEquals(data(20.0, 20.0, 1.0, 1.0, 12.0, 12.0), x82.getGradient());

    VariableTensor x91 = matrix(5, 4).variable().withData(-2.0, 5.0, -3.0, 3.0, 10.0, -8.0, -2.0,
        8.0, -10.0, 8.0, 9.0, -3.0, 2.0, 10.0, -1.0, 2.0, 1.0, 9.0, 5.0, 8.0);
    VariableTensor x92 = matrix(4, 1).variable().withData(1.0, 7.0, 0.0, 9.0);
    ComputedTensor y9 = matmul(x91, x92);
    assertTensorsEquals(matrix(5, 1).withData(60.0, 26.0, 19.0, 90.0, 136.0), y9);
    y9.backpropagate(matrix(5, 1).ones().getData());
    System.out.println(Arrays.toString(x91.getGradient()));
    System.out.println(Arrays.toString(x92.getGradient()));
    assertArrayEquals(data(1.0, 7.0, 0.0, 9.0, 1.0, 7.0, 0.0, 9.0, 1.0, 7.0, 0.0, 9.0, 1.0, 7.0,
        0.0, 9.0, 1.0, 7.0, 0.0, 9.0), x91.getGradient());
    assertArrayEquals(data(1.0, 24.0, 8.0, 18.0), x92.getGradient());

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
