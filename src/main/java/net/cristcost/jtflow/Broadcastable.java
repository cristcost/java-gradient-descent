package net.cristcost.jtflow;

import java.util.Arrays;

public interface Broadcastable extends Tensor {

  default Tensor broadcast(final int... shape) {

    return new Tensor() {
      final int broadcastSize = Arrays.stream(shape).reduce(1, (a, b) -> a * b);

      public int size() {
        return broadcastSize;
      }

      @Override
      public int[] getShape() {
        return shape;
      }

      @Override
      public double[] getData() {
        return Broadcastable.this.getData();
      }
    };
  }
}
