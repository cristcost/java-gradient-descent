package net.cristcost.jtflow.builder;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.random.RandomGenerator;
import net.cristcost.jtflow.api.Optimizer;
import net.cristcost.jtflow.api.Tensor;
import net.cristcost.jtflow.tensors.ConstantTensor;
import net.cristcost.jtflow.tensors.VariableTensor;


public class TensorBuilder<T extends Tensor> {



  public static TensorBuilder<ConstantTensor> builder(int[] shape) {
    return new TensorBuilder<>(ConstantTensor.class, shape, (d, s) -> new ConstantTensor(d, s));
  }

  static void incrementIndices(int[] indices, int[] shape) {
    int cursor = indices.length - 1;
    indices[cursor]++;
    while (cursor >= 0 && indices[cursor] >= shape[cursor]) {
      indices[cursor] = 0;
      cursor--;
      if (cursor < 0) {
        break;
      }
      indices[cursor]++;
    }
  }

  private static int shapeSize(int[] s) {
    return Arrays.stream(s).reduce(1, (a, b) -> a * b);
  }

  private final BiFunction<double[], int[], T> buildFunction;

  private final int[] shape;

  private final Class<T> tensorType;

  private TensorBuilder<VariableTensor> variableTensorBuilder;

  private Optimizer optimizer;

  private TensorBuilder(Class<T> tensorType, int[] shape,
      BiFunction<double[], int[], T> buildFunction) {
    this.tensorType = tensorType;
    this.shape = shape;
    this.buildFunction = buildFunction;
  }

  public T clone(Tensor tensor) {
    if (shapeSize(shape) != shapeSize(tensor.getShape())) {
      throw new RuntimeException(
          String.format("Clone tensor failed, tensor shape %s incompatible with builder shape s",
              Arrays.toString(tensor.getShape()),
              Arrays.toString(shape)));
    }

    return withData(tensor.getData().clone());
  }

  public T kaimingUniform() {
    double inputFeatures = shape[0];
    double maxval = Math.sqrt(1.0 / (inputFeatures));

    return rand(() -> RandomGenerator.getDefault().nextDouble(-maxval, maxval));
  }

  public T normal(double mean, double standarDeviation) {
    return rand(() -> RandomGenerator.getDefault().nextGaussian(mean, standarDeviation));
  }


  public T ones() {
    int size = shapeSize(shape);
    double[] data = new double[size];
    Arrays.fill(data, 1.0);
    return withData(data);
  }

  public T rand(Supplier<Double> randomFunction) {
    int size = shapeSize(shape);
    double[] data = new double[size];
    for (int i = 0; i < size; i++) {
      data[i] = randomFunction.get();
    }
    return withData(data);
  }

  public T repeat(double value) {
    int size = shapeSize(shape);
    double[] data = new double[size];
    Arrays.fill(data, value);
    return withData(data);
  }

  public T uniform(double minval, double maxval) {
    return rand(() -> RandomGenerator.getDefault().nextDouble(minval, maxval));
  }

  @SuppressWarnings("unchecked")
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

  public T withData(BiFunction<Integer, Integer, Double> dataSupplier) {
    return withData((i) -> dataSupplier.apply(i[0], i[1]));
  }

  public T withData(double... data) {
    return buildFunction.apply(data, shape);
  }


  public T withData(Function<int[], Double> dataSupplier) {
    double[] data = new double[shapeSize(shape)];

    int[] ii = new int[shape.length];
    for (int i = 0; i < data.length; i++) {
      data[i] = dataSupplier.apply(ii);

      // increment indices
      incrementIndices(ii, shape);
    }

    return buildFunction.apply(data, shape);
  }

  public T zeros() {
    int size = shapeSize(shape);
    double[] data = new double[size];
    Arrays.fill(data, 0.0);
    return withData(data);
  }

  TensorBuilder<T> withOptimizer(Optimizer optimizer) {
    this.optimizer = optimizer;
    return this;
  }
}
