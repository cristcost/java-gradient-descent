package net.cristcost.jtflow.api;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import java.util.function.Function;

/**
 * Tensor interface.
 * 
 * Represent a tensor type for the custom math library.
 */
public interface Tensor {

  public static Tensor fromFile(Path file) throws IOException {

    try (DataInputStream dis = new DataInputStream(Files.newInputStream(file))) {
      int shapeSize = dis.readInt();
      int dataSize = dis.readInt();

      final int[] shape = new int[shapeSize];
      for (int i = 0; i < shapeSize; i++) {
        shape[i] = dis.readInt();
      }

      final double[] data = new double[dataSize];
      for (int i = 0; i < dataSize; i++) {
        data[i] = dis.readDouble();
      }

      return new Tensor() {
        @Override
        public double[] getData() {
          return data;
        }

        @Override
        public int[] getShape() {
          return shape;
        }
      };
    }
  }

  public static void incrementIndices(int[] indices, int[] shape) {
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

  static int calculateIndex(int[] shape, int[] indices) {
    int indicesNdim = indices.length;
    if (indicesNdim == 0) {
      return 0;
    } else if (indicesNdim == 1) {
      return indices[0];
    } else {

      int tensorNdim = shape.length;

      if (indicesNdim > tensorNdim) {
        throw new IllegalArgumentException("Number of indices does not match array dimension.");
      } else {
        int index = 0;
        int multiplier = 1;
        for (int i = tensorNdim - 1; i >= (tensorNdim - indicesNdim); i--) {
          // shift the requested indices to the right by subtracting (tensorNdim - indicesNdim)
          index += indices[i - tensorNdim + indicesNdim] * multiplier;
          multiplier *= shape[i];
        }
        return index;
      }
    }
  }

  default <T> Optional<T> broadcastable(Function<Broadcastable, T> function) {
    if (this instanceof Broadcastable) {
      Broadcastable broadcastable = (Broadcastable) this;
      return Optional.of(function.apply(broadcastable));
    } else {
      return Optional.empty();
    }
  }

  default double get(int... indices) {
    int index = calculateIndex(indices);
    if (index >= size()) {
      throw new ArrayIndexOutOfBoundsException(
          String.format(
              "Requested index is beyond the size of the tensor data: result index %d >= size %d",
              index, size()));
    }
    return getData()[index % getData().length];
  }

  double[] getData();

  int[] getShape();

  default String json() {
    return json(-1);
  }

  default String json(int decimals) {
    StringBuilder builder = new StringBuilder();
    if (getShape().length == 0) {
      // Scalar with no shape
      builder.append(getData()[0]);
    } else {
      formatArray(decimals, builder, 0, getData().length, 0);
    }
    return builder.toString();
  }

  default int size() {
    return getData().length;
  }

  default void toFile(Path file) throws IOException {
    try (DataOutputStream dos = new DataOutputStream(Files.newOutputStream(file))) {
      dos.writeInt(getShape().length);
      dos.writeInt(getData().length);
      for (int i : getShape()) {
        dos.writeInt(i);
      }

      for (double d : getData()) {
        dos.writeDouble(d);
      }
    }
  }

  private int calculateIndex(int[] indices) {
    return calculateIndex(getShape(), indices);
  }


  private void formatArray(int decimals, StringBuilder builder, int index, int size, int level) {
    if (getShape().length - level == 1) {
      builder.append("[");
      for (int i = index; i < size; i++) {
        if (i > index) {
          builder.append(", ");
        }
        builder.append(
            decimals >= 0 ? String.format("%." + decimals + "f", getData()[i]) : getData()[i]);
      }
      builder.append("]");
    } else {

      int stride = 1;
      for (int i = 1; i < getShape().length - level; i++) {
        stride *= getShape()[i];
      }
      builder.append("[");
      for (int i = 0; i < getShape()[0]; i++) {
        if (i > 0) {
          builder.append(", ");
        }
        formatArray(decimals, builder, index, index + stride, level + 1);
        index += stride;
      }
      builder.append("]");
    }
  }
}
