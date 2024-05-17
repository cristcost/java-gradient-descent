package net.cristcost.differentiable;

import static net.cristcost.differentiable.MathLibrary.*;
import static net.cristcost.differentiable.NDimensionalArray.ndscalar;
import static org.junit.jupiter.api.Assertions.*;
import java.util.function.Function;
import org.junit.jupiter.api.Test;

class MathLibraryTest {


  @Test
  void testSum() {
    assertEquals(ndscalar(3.0 + 5.0), sum(constant(3.0), constant(5.0)).getValue());
    assertEquals(ndscalar(7.0 - 12.0 + 18.0),
        sum(constant(7.0), constant(-12.0), constant(18.0)).getValue());
    
    assertEquals(ndscalar(3.0 + 5.0), sum(constant(3.0), constant(5.0)).getValue());
    
    
  }

  @Test
  void testMultiply() {
    assertEquals(ndscalar(3.0 * 5.0), multiply(constant(3.0), constant(5.0)).getValue());
    assertEquals(ndscalar(0.5 * -0.3 * -1.0),
        multiply(constant(0.5), constant(-0.3), constant(-1.0)).getValue());
  }

  @Test
  void testPow() {
    assertEquals(ndscalar(Math.pow(2.0, 8.0)), pow(constant(2.0), constant(8.0)).getValue());
    assertEquals(ndscalar(Math.pow(4.0, 0.5)), pow(constant(4.0), constant(0.5)).getValue());
    assertEquals(ndscalar(Math.pow(3.0, -1.0)), pow(constant(3.0), constant(-1.0)).getValue());
  }

  @Test
  void testRelu() {
    assertEquals(ndscalar(3.0), relu(constant(3.0)).getValue());
    assertEquals(ndscalar(0.0), relu(constant(-5.0)).getValue());
  }

  @Test
  void testCombinedOperation() {

    // f(x) = relu(0.5x^2 - 2x -6)

    Function<Tensor, Tensor> f = x -> relu(
        sum(
            multiply((constant(-0.5)), pow(x, constant(2.0))),
            multiply(constant(2.0), x),
            constant(6.0)));

    assertEquals(ndscalar(0.0), f.apply(constant(-4.0)).getValue());
    assertEquals(ndscalar(0.0), f.apply(constant(-2.0)).getValue());
    assertEquals(ndscalar(6.0), f.apply(constant(0.0)).getValue());
    assertEquals(ndscalar(8.0), f.apply(constant(2.0)).getValue());
    assertEquals(ndscalar(6.0), f.apply(constant(4.0)).getValue());
    assertEquals(ndscalar(0.0), f.apply(constant(6.0)).getValue());
    assertEquals(ndscalar(0.0), f.apply(constant(8.0)).getValue());

  }

}
