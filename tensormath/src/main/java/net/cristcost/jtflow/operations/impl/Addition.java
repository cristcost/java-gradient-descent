package net.cristcost.jtflow.operations.impl;

import net.cristcost.jtflow.api.Tensor;

public class Addition {

  public static void chain(double[] outerFunctionGradient, Tensor... operands) {
    for (Tensor operand : operands) {
      operand.ifChainable(c -> c.backpropagate(outerFunctionGradient));
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
