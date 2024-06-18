package net.cristcost.jtflow.builder;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import net.cristcost.jtflow.builder.TensorBuilder;

class TensorIndicesTest {

  @Test
  void test() {
    int[] shape = shape(3, 3, 3);
    int[] indices = new int[3];

    Assertions.assertArrayEquals(index(0, 0, 0), indices);

    TensorBuilder.incrementIndices(indices, shape);
    Assertions.assertArrayEquals(index(0, 0, 1), indices);

    TensorBuilder.incrementIndices(indices, shape);
    Assertions.assertArrayEquals(index(0, 0, 2), indices);

    TensorBuilder.incrementIndices(indices, shape);
    Assertions.assertArrayEquals(index(0, 1, 0), indices);

    repeat(3, () -> TensorBuilder.incrementIndices(indices, shape));
    Assertions.assertArrayEquals(index(0, 2, 0), indices);

    repeat(3, () -> TensorBuilder.incrementIndices(indices, shape));
    Assertions.assertArrayEquals(index(1, 0, 0), indices);


    repeat(8, () -> TensorBuilder.incrementIndices(indices, shape));
    Assertions.assertArrayEquals(index(1, 2, 2), indices);

    TensorBuilder.incrementIndices(indices, shape);
    Assertions.assertArrayEquals(index(2, 0, 0), indices);

    repeat(8, () -> TensorBuilder.incrementIndices(indices, shape));
    Assertions.assertArrayEquals(index(2, 2, 2), indices);

    TensorBuilder.incrementIndices(indices, shape);
    Assertions.assertArrayEquals(index(0, 0, 0), indices);

  }

  private int[] index(int... indices) {
    return indices;
  }

  private int[] shape(int... shape) {
    return shape;
  }

  private void repeat(int times, Runnable runnable) {
    for (int i = 0; i < times; i++) {
      runnable.run();
    }
  }
}
