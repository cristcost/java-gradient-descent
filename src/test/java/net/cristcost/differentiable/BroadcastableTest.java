package net.cristcost.differentiable;

import static net.cristcost.differentiable.MathLibrary.constant;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;

class BroadcastableTest {

  @Test
  void testBroadcastMatrix() {
    var builder = constant(3, 3);

    Tensor tensor = builder.withData(1, 2, 3, 4, 5, 6, 7, 8, 9);

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

    // overflowing the whole data
    assertEquals(1.0, tensor.get(0, 9));
    assertEquals(1.0, tensor.get(0, 9 * 2));
    assertEquals(1.0, tensor.get(0, 9 * 10));

    assertEquals(1.0, tensor.get(0, 0));
    assertEquals(1.0, tensor.get(3, 0));
    assertEquals(1.0, tensor.get(3 * 10, 0));


    // bigger indexes
    assertEquals(1.0, tensor.broadcastable(b -> b.broadcast(1, 3, 3)).get().get(0, 0, 0));
    assertEquals(5.0, tensor.broadcastable(b -> b.broadcast(1, 1, 3, 3)).get().get(0, 0, 1, 1));
    assertEquals(8.0,
        tensor.broadcastable(b -> b.broadcast(1, 1, 1, 3, 3)).get().get(0, 0, 0, 2, 1));
  }

}
