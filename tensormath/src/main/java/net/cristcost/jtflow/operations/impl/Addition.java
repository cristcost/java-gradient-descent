package net.cristcost.jtflow.operations.impl;

import net.cristcost.jtflow.api.Chainable;
import net.cristcost.jtflow.api.Tensor;

// Not tested and not to be used in this form
@Deprecated()
public class Addition {

  public static void chain(double[] outerFunctionGradient, Tensor... operands) {
    for (Tensor operand : operands) {
      if (operand instanceof Chainable) {
        ((Chainable) operand).backpropagate(outerFunctionGradient);
      }
    }
  }

  public static double[] compute(Tensor... operands) {
    double[] data = new double[operands[0].size()];

    for (Tensor t : operands) {
      if (data.length != t.size()) {
        throw new IllegalArgumentException("Shapes do not match.");
      }
      for (int i = 0; i < data.length; i++) {
        data[i] += t.get(i);
      }
    }
    return data;
  }

}
