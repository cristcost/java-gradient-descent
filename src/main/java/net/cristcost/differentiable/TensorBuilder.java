package net.cristcost.differentiable;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Supplier;
import lombok.RequiredArgsConstructor;


@RequiredArgsConstructor
public class TensorBuilder {

  private final int[] shape;

  private final BiFunction<double[], int[], Tensor> buildFunction;


  public Tensor withData(double... data) {
    return buildFunction.apply(data, shape);
  }

  public Tensor repeat(double value) {
    int size = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
    double[] data = new double[size];
    Arrays.fill(data, value);
    return withData(data);
  }

  public Tensor zeros() {
    int size = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
    double[] data = new double[size];
    Arrays.fill(data, 0.0);
    return withData(data);
  }

  public Tensor ones() {
    int size = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
    double[] data = new double[size];
    Arrays.fill(data, 1.0);
    return withData(data);
  }

  public Tensor rand(Supplier<Double> randomFunction) {
    int size = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
    double[] data = new double[size];
    for (int i = 0; i < size; i++) {
      data[i] = randomFunction.get();
    }
    return withData(data);
  }
}
