package net.cristcost.differentiable;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class MathLibraryTransposeTest {

  @Test
  void testTranspose() {
    ConstantTensor tensor = MathLibrary.matrix(2, 5).withData((i, j) -> 1.0 + j + 5.0 * i);

    Tensor transpose = MathLibrary.transpose(tensor);

    int[] shape = tensor.getShape();
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        // System.out.println(tensor.get(j, i));
        Assertions.assertEquals(tensor.get(i, j), transpose.get(j, i));
      }
    }

  }
}
