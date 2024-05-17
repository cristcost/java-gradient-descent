package net.cristcost.differentiable;

import java.util.Arrays;
import lombok.Getter;

public class NDimensionalArray {

  @Getter
  private double[] data;

  @Getter
  private int[] shape;

  public static double[] data(double... data) {
    return data;
  }

  public static int[] shape(int... shape) {
    return shape;
  }

  public static NDimensionalArray zeros_like(NDimensionalArray ref) {
    NDimensionalArray result = new NDimensionalArray(ref.shape);
    Arrays.fill(result.data, 0.0);
    return result;
  }

  public static NDimensionalArray ones_like(NDimensionalArray ref) {
    NDimensionalArray result = new NDimensionalArray(ref.shape);
    Arrays.fill(result.data, 1.0);
    return result;
  }

  public NDimensionalArray(int[] shape) {
    this.shape = shape;
    int size = 1;
    for (int dim : shape) {
      size *= dim;
    }
    this.data = new double[size];
  }

  public static NDimensionalArray ndscalar(double value) {
    return new NDimensionalArray(data(value), shape());
  }

  public static NDimensionalArray ndvector(double... value) {
    return new NDimensionalArray(value, shape(value.length));
  }

  public static NDimensionalArray ndmatrix(double[] value, int rows, int columns) {
    if (value.length != rows * columns) {
      throw new IllegalArgumentException("Data length does not match shape.");
    }
    return new NDimensionalArray(value, shape(rows, columns));
  }

  public NDimensionalArray(double[] data) {
    this.data = data;
    this.shape = shape(data.length);
  }

  public NDimensionalArray(double[] data, int... shape) {
    if (data.length != calculateSize(shape)) {
      throw new IllegalArgumentException("Data length does not match shape.");
    }
    this.data = data;
    this.shape = shape;
  }


  public double get(int... indices) {
    int index = calculateIndex(indices);
    return data[index];
  }

  public void set(double value, int... indices) {
    int index = calculateIndex(indices);
    data[index] = value;
  }

  public NDimensionalArray add(NDimensionalArray other) {
    if (!Arrays.equals(this.shape, other.shape)) {
      throw new IllegalArgumentException("Shapes do not match.");
    }
    double[] resultData = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      resultData[i] = this.data[i] + other.data[i];
    }
    return new NDimensionalArray(resultData, shape);
  }

  public NDimensionalArray subtract(NDimensionalArray other) {
    if (!Arrays.equals(this.shape, other.shape)) {
      throw new IllegalArgumentException("Shapes do not match.");
    }
    double[] resultData = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      resultData[i] = this.data[i] - other.data[i];
    }
    return new NDimensionalArray(resultData, shape);
  }

  public NDimensionalArray multiply(double scalar) {
    double[] resultData = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      resultData[i] = this.data[i] * scalar;
    }
    return new NDimensionalArray(resultData, shape);
  }

  private int calculateSize(int[] shape) {
    int size = 1;
    for (int dim : shape) {
      size *= dim;
    }
    return size;
  }

  private int calculateIndex(int... indices) {
    if (indices.length != shape.length) {
      throw new IllegalArgumentException("Number of indices does not match array dimension.");
    }
    int index = 0;
    int multiplier = 1;
    for (int i = shape.length - 1; i >= 0; i--) {
      if (indices[i] >= shape[i] || indices[i] < 0) {
        throw new IllegalArgumentException("Index out of bounds.");
      }
      index += indices[i] * multiplier;
      multiplier *= shape[i];
    }
    return index;
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + Arrays.hashCode(data);
    result = prime * result + Arrays.hashCode(shape);
    return result;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    NDimensionalArray other = (NDimensionalArray) obj;
    return Arrays.equals(shape, other.shape) && Arrays.equals(data, other.data);
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    formatArray(builder, 0, data.length, 0);
    return builder.toString();
  }

  private void formatArray(StringBuilder builder, int index, int size, int level) {
    if (shape.length == 0) {
      // Scalar with no shape
      builder.append(data[0]);
    } else if (shape.length - level == 1) {
      builder.append("[");
      for (int i = index; i < size; i++) {
        if (i > index) {
          builder.append(", ");
        }
        builder.append(data[i]);
      }
      builder.append("]");
    } else {

      int stride = 1;
      for (int i = 1; i < shape.length - level; i++) {
        stride *= shape[i];
      }
      builder.append("[");
      for (int i = 0; i < shape[0]; i++) {
        if (i > 0) {
          builder.append(", ");
        }
        formatArray(builder, index, index + stride, level + 1);
        index += stride;
      }
      builder.append("]");
    }
  }
}
