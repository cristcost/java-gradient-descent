package net.cristcost.differentiable;

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
    return getData()[index];
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


  private int calculateIndex(int[] indices) {
    if (indices.length == 0) {
      return 0;
    } else if (indices.length == 1) {
      return indices[0];
    } else if (indices.length > getShape().length) {
      throw new IllegalArgumentException("Number of indices does not match array dimension.");
    } else {
      int index = 0;
      int multiplier = 1;
      for (int i = getShape().length - 1; i >= (getShape().length - indices.length); i--) {
        index += indices[i] * multiplier;
        multiplier *= getShape()[i];
      }
      return index;
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
