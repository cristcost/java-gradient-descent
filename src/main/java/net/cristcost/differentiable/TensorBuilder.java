package net.cristcost.differentiable;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Supplier;


public class TensorBuilder<T extends Tensor> {

  private final int[] shape;

  private final BiFunction<double[], int[], T> buildFunction;

  private TensorBuilder(int[] shape, BiFunction<double[], int[], T> buildFunction) {
    this.shape = shape;
    this.buildFunction = buildFunction;
  }

  public T withData(double... data) {
    return buildFunction.apply(data, shape);
  }

  public static TensorBuilder<ConstantTensor> constant(int[] shape) {
    return new TensorBuilder<>(shape, (d, s) -> new ConstantTensor(d, s));
  }
  
  public TensorBuilder<ConstantTensor> constant() {
    return new TensorBuilder<>(this.shape, (d, s) -> new ConstantTensor(d, s));
  }

  public TensorBuilder<VariableTensor> variable() {
    return new TensorBuilder<>(this.shape, (d, s) -> new VariableTensor(d, s));
  }


  public T repeat(double value) {
    int size = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
    double[] data = new double[size];
    Arrays.fill(data, value);
    return withData(data);
  }

  public T zeros() {
    int size = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
    double[] data = new double[size];
    Arrays.fill(data, 0.0);
    return withData(data);
  }

  public T ones() {
    int size = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
    double[] data = new double[size];
    Arrays.fill(data, 1.0);
    return withData(data);
  }

  public T rand(Supplier<Double> randomFunction) {
    int size = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
    double[] data = new double[size];
    for (int i = 0; i < size; i++) {
      data[i] = randomFunction.get();
    }
    return withData(data);
  }
}
