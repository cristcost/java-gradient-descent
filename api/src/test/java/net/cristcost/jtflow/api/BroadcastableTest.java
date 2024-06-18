package net.cristcost.jtflow.api;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class BroadcastableTest {

  @Test
  void testBroadcastMatrix() {

    final double[] data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    final int[] shape = {3, 3};
    class TestTensor implements Tensor, Broadcastable {
      @Override
      public double[] getData() {
        return data;
      }

      @Override
      public int[] getShape() {
        return shape;
      }
    }
    Tensor tensor = new TestTensor();

    // correct indexing
    assertEquals(1.0, tensor.get(0, 0));
    assertEquals(2.0, tensor.get(0, 1));
    assertEquals(3.0, tensor.get(0, 2));
    assertEquals(4.0, tensor.get(1, 0));
    assertEquals(5.0, tensor.get(1, 1));
    assertEquals(6.0, tensor.get(1, 2));
    assertEquals(7.0, tensor.get(2, 0));
    assertEquals(8.0, tensor.get(2, 1));
    assertEquals(9.0, tensor.get(2, 2));

    // overflowing an index
    assertEquals(4.0, tensor.get(0, 3)); // overflow to next row
    assertEquals(5.0, tensor.get(0, 4));
    assertEquals(6.0, tensor.get(0, 5));

    assertEquals(7.0, tensor.get(0, 6));
    assertEquals(8.0, tensor.get(0, 7));
    assertEquals(9.0, tensor.get(0, 8));

    assertEquals(7.0, tensor.get(1, 3)); // overflow to next row
    assertEquals(8.0, tensor.get(1, 4));
    assertEquals(9.0, tensor.get(1, 5));

    // last dimension indexing
    assertEquals(1.0, tensor.get(0));
    assertEquals(2.0, tensor.get(1));
    assertEquals(3.0, tensor.get(2));
    assertEquals(4.0, tensor.get(3));
    assertEquals(5.0, tensor.get(4));
    assertEquals(6.0, tensor.get(5));
    assertEquals(7.0, tensor.get(6));
    assertEquals(8.0, tensor.get(7));
    assertEquals(9.0, tensor.get(8));

    // overflowing the data
    assertThrows(ArrayIndexOutOfBoundsException.class, () -> tensor.get(0, 9));
    assertThrows(ArrayIndexOutOfBoundsException.class, () -> tensor.get(0, 9 * 2));
    assertThrows(ArrayIndexOutOfBoundsException.class, () -> tensor.get(0, 9 * 10));

    assertThrows(ArrayIndexOutOfBoundsException.class, () -> tensor.get(3, 0));
    assertThrows(ArrayIndexOutOfBoundsException.class, () -> tensor.get(3 * 10, 0));

    assertThrows(ArrayIndexOutOfBoundsException.class, () -> tensor.get(9));

    // broadcast to more dimensions
    assertEquals(1.0, tensor.broadcastable(b -> b.broadcast(1, 3, 3)).get().get(0, 0, 0));
    assertEquals(5.0, tensor.broadcastable(b -> b.broadcast(1, 1, 3, 3)).get().get(0, 0, 1, 1));
    assertEquals(8.0,
        tensor.broadcastable(b -> b.broadcast(1, 1, 1, 3, 3)).get().get(0, 0, 0, 2, 1));

    // broadcast to more dimensions and overflow (these fail in previous test)
    assertEquals(1.0, tensor.broadcastable(b -> b.broadcast(2, 3, 3)).get().get(0, 9));
    assertEquals(1.0, tensor.broadcastable(b -> b.broadcast(3, 3, 3)).get().get(0, 9 * 2));
    assertEquals(1.0, tensor.broadcastable(b -> b.broadcast(11, 3, 3)).get().get(0, 9 * 10));

    assertEquals(1.0, tensor.broadcastable(b -> b.broadcast(2, 3, 3)).get().get(3, 0));
    assertEquals(1.0, tensor.broadcastable(b -> b.broadcast(11, 3, 3)).get().get(3 * 10, 0));
  }

}
