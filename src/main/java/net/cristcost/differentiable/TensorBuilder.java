package net.cristcost.differentiable;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Supplier;
import java.util.random.RandomGenerator;


public class TensorBuilder<T extends Tensor> {

  private final int[] shape;

  private final BiFunction<double[], int[], T> buildFunction;

  private final Class<T> tensorType;

  private TensorBuilder<VariableTensor> variableTensorBuilder;

  private TensorBuilder(Class<T> tensorType, int[] shape,
      BiFunction<double[], int[], T> buildFunction) {
    this.tensorType = tensorType;
    this.shape = shape;
    this.buildFunction = buildFunction;
  }

  public T withData(double... data) {
    return buildFunction.apply(data, shape);
  }

  public static TensorBuilder<ConstantTensor> builder(int[] shape) {
    return new TensorBuilder<>(ConstantTensor.class, shape, (d, s) -> new ConstantTensor(d, s));
  }

  public TensorBuilder<VariableTensor> variable() {
    if (tensorType == VariableTensor.class) {
      // Reuse himself if by chance this is called twice
      return (TensorBuilder<VariableTensor>) this;
    } else if (variableTensorBuilder == null) {
      // Reuse the variableTensorBuilder as shape is final and can't change
      variableTensorBuilder = new TensorBuilder<>(VariableTensor.class, this.shape,
          (d, s) -> new VariableTensor(d, s));
    }
    return variableTensorBuilder;
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

  public T normal(double mean, double standarDeviation) {
    return rand(() -> RandomGenerator.getDefault().nextGaussian(mean, standarDeviation));
  }

  public T uniform(double minval, double maxval) {
    return rand(() -> RandomGenerator.getDefault().nextDouble(minval, minval));
  }
}
