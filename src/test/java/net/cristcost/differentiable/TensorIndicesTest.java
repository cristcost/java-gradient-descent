package net.cristcost.differentiable;

import static net.cristcost.jtflow.MathLibrary.*;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import net.cristcost.jtflow.Tensor;

class TensorIndicesTest {

  @Test
  void test() {
    int[] shape = shape(3, 3, 3);
    int[] indices = new int[3];

    Assertions.assertArrayEquals(index(0, 0, 0), indices);

    Tensor.incrementIndices(indices, shape);
    Assertions.assertArrayEquals(index(0, 0, 1), indices);

    Tensor.incrementIndices(indices, shape);
    Assertions.assertArrayEquals(index(0, 0, 2), indices);

    Tensor.incrementIndices(indices, shape);
    Assertions.assertArrayEquals(index(0, 1, 0), indices);

    repeat(3, () -> Tensor.incrementIndices(indices, shape));
    Assertions.assertArrayEquals(index(0, 2, 0), indices);

    repeat(3, () -> Tensor.incrementIndices(indices, shape));
    Assertions.assertArrayEquals(index(1, 0, 0), indices);


    repeat(8, () -> Tensor.incrementIndices(indices, shape));
    Assertions.assertArrayEquals(index(1, 2, 2), indices);

    Tensor.incrementIndices(indices, shape);
    Assertions.assertArrayEquals(index(2, 0, 0), indices);

    repeat(8, () -> Tensor.incrementIndices(indices, shape));
    Assertions.assertArrayEquals(index(2, 2, 2), indices);
    
    Tensor.incrementIndices(indices, shape);
    Assertions.assertArrayEquals(index(0, 0, 0), indices);

  }

  private void repeat(int times, Runnable runnable) {
    for (int i = 0; i < times; i++) {
      runnable.run();
    }
  }
}
