package net.cristcost.differentiable;

import static net.cristcost.differentiable.MathLibrary.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import java.util.function.Function;
import org.junit.jupiter.api.Test;

class MathLibraryBackpropagationTest {

  @Test
  void testSum() {
    VariableScalar x1 = variable(3.0);
    ComputedScalar y1 = sum(x1, constant(5.0));
    y1.startBackpropagation();
    assertEquals(1.0, x1.getGradient());

    VariableScalar x21 = variable(7.0);
    VariableScalar x22 = variable(-12.0);
    ComputedScalar y2 = sum(x21, x22, constant(18.0));
    y2.startBackpropagation();
    assertEquals(1.0, x21.getGradient());
    assertEquals(1.0, x22.getGradient());

    VariableScalar x3 = variable(5.0);
    ComputedScalar y3 = sum(x3, x3, x3);
    y3.startBackpropagation();
    assertEquals(3.0, x3.getGradient());

  }

  @Test
  void testMultiply() {
    VariableScalar x1 = variable(3.0);
    ComputedScalar y1 = multiply(x1, constant(5.0));
    y1.startBackpropagation();
    assertEquals(5.0, x1.getGradient());

    VariableScalar x21 = variable(0.5);
    VariableScalar x22 = variable(-0.3);
    ComputedScalar y2 = multiply(x21, x22, constant(-1.0));
    y2.startBackpropagation();
    assertEquals(x22.getValue() * -1.0, x21.getGradient());
    assertEquals(x21.getValue() * -1.0, x22.getGradient());

    VariableScalar x3 = variable(2.0);
    ComputedScalar y3 = multiply(x3, x3, x3);
    y3.startBackpropagation();
    assertEquals(2.0 * 2.0 * 3.0, x3.getGradient());
  }

  @Test
  void testBasicPowerOperation() {

    VariableScalar x1 = variable(2.0);
    ComputedScalar y1 = pow(x1, constant(8.0));
    y1.startBackpropagation();
    assertEquals(8.0 * Math.pow(2.0, 7.0), x1.getGradient());


    VariableScalar x2 = variable(3.0);
    ComputedScalar y2 = pow(constant(Math.E), x2);
    y2.startBackpropagation();
    assertEquals(Math.pow(Math.E, 3.0), x2.getGradient());

    VariableScalar x3 = variable(0.5);
    ComputedScalar y3 = pow(constant(4.0), x3);
    y3.startBackpropagation();
    assertEquals(Math.log(4.0) * Math.pow(4.0, 0.5), x3.getGradient());

    VariableScalar x41 = variable(5.0);
    VariableScalar x42 = variable(3.0);
    ComputedScalar y4 = pow(x41, x42);
    y4.startBackpropagation();
    assertEquals(3.0 * Math.pow(5.0, 2.0), x41.getGradient());
    assertEquals(Math.log(5.0) * Math.pow(5.0, 3.0), x42.getGradient());

    VariableScalar x5 = variable(3.0);
    ComputedScalar y5 = pow(x5, x5);
    y5.startBackpropagation();
    assertEquals((Math.log(3.0) + 1.0) * Math.pow(3.0, 3.0), x5.getGradient());

  }

  @Test
  void testBasicReluOperation() {
    VariableScalar x1 = variable(3.0);
    ComputedScalar y1 = relu(x1);
    y1.startBackpropagation();
    assertEquals(1.0, x1.getGradient());


    VariableScalar x2 = variable(-3.0);
    ComputedScalar y2 = relu(x2);
    y2.startBackpropagation();
    assertEquals(0.0, x2.getGradient());
  }

  @Test
  void testComplexOperation() {

    // f(x) = relu(-0.5x^2 + 2x + 6)
    // df(x) = drelu(-0.5x^2 + 2x + 6) * d(-0.5x^2 + 2x + 6) = drelu(-0.5x^2 + 2x +6) * (-x + 2)

    Function<VariableScalar, VariableScalar> df = x -> {
      ComputedScalar y = relu(sum(
          multiply((constant(-0.5)), pow(x, constant(2.0))),
          multiply(constant(2.0), x),
          constant(6.0)));
      y.startBackpropagation();
      return x;
    };


    assertEquals(0.0, df.apply(variable(-4.0)).getGradient());
    assertEquals(0.0, df.apply(variable(-2.0)).getGradient());
    assertEquals(2.0, df.apply(variable(0.0)).getGradient());
    assertEquals(0.0, df.apply(variable(2.0)).getGradient());
    assertEquals(-2.0, df.apply(variable(4.0)).getGradient());
    assertEquals(0.0, df.apply(variable(6.0)).getGradient());
    assertEquals(0.0, df.apply(variable(8.0)).getGradient());

  }

}
