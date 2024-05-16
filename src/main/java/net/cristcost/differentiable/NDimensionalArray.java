package net.cristcost.differentiable;

import java.util.Arrays;

public class NDimensionalArray {
  private double[] data;
  private int[] shape;

  public NDimensionalArray(int... shape) {
    this.shape = shape;
    int size = 1;
    for (int dim : shape) {
      size *= dim;
    }
    this.data = new double[size];
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
}
