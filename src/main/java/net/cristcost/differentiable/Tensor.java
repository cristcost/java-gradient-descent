package net.cristcost.differentiable;

import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Tensor interface.
 * 
 * Represent a tensor type for the custom math library.
 */
public interface Tensor {

  double[] getData();

  int[] getShape();

  default int size() {
    return getData().length;
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

  default String json() {
    StringBuilder builder = new StringBuilder();
    if (getShape().length == 0) {
      // Scalar with no shape
      builder.append(getData()[0]);
    } else {
      formatArray(builder, 0, getData().length, 0);
    }
    return builder.toString();
  }

  default <T> Optional<T> broadcastable(Function<Broadcastable, T> function) {
    if (this instanceof Broadcastable) {
      Broadcastable broadcastable = (Broadcastable) this;
      return Optional.of(function.apply(broadcastable));
    } else {
      return Optional.empty();
    }
  }

  private int calculateIndex(int[] indices) {
    int indicesNdim = indices.length;
    if (indicesNdim == 0) {
      return 0;
    } else if (indicesNdim == 1) {
      return indices[0];
    } else {

      int tensorNdim = getShape().length;

      if (indicesNdim > tensorNdim) {
        throw new IllegalArgumentException("Number of indices does not match array dimension.");
      } else {
        int index = 0;
        int multiplier = 1;
        for (int i = tensorNdim - 1; i >= (tensorNdim - indicesNdim); i--) {
          // shift the requested indices to the right by subtracting (tensorNdim - indicesNdim)
          index += indices[i - tensorNdim + indicesNdim] * multiplier;
          multiplier *= getShape()[i];
        }
        return index;
      }
    }
  }


  private void formatArray(StringBuilder builder, int index, int size, int level) {
    if (getShape().length - level == 1) {
      builder.append("[");
      for (int i = index; i < size; i++) {
        if (i > index) {
          builder.append(", ");
        }
        builder.append(getData()[i]);
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
        formatArray(builder, index, index + stride, level + 1);
        index += stride;
      }
      builder.append("]");
    }
  }
}
