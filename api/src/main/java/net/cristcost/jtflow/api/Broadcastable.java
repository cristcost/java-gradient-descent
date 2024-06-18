package net.cristcost.jtflow.api;

import java.util.Arrays;

public interface Broadcastable extends Tensor {

  default Tensor broadcast(final int... shape) {

    return new Tensor() {
      final int broadcastSize = Arrays.stream(shape).reduce(1, (a, b) -> a * b);

      @Override
      public double[] getData() {
        return Broadcastable.this.getData();
      }

      @Override
      public int[] getShape() {
        return shape;
      }

      public int size() {
        return broadcastSize;
      }
    };
  }
}
