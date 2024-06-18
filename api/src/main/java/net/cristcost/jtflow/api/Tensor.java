package net.cristcost.jtflow.api;

/**
 * Tensor interface.
 * 
 * Represent a tensor type for the custom math library.
 */
public interface Tensor extends Projector {



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

  default int calculateIndex(int[] indices) {
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
