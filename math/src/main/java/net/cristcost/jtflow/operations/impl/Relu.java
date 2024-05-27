package net.cristcost.jtflow.operations.impl;

import net.cristcost.jtflow.api.Chainable;
import net.cristcost.jtflow.api.Tensor;
//Not tested and not to be used in this form 
@Deprecated()
public class Relu {

  public static void relu(double[] outerFunctionGradient, Tensor operand) {
    if (operand instanceof Chainable) {
      double[] innerGradient = new double[outerFunctionGradient.length];
      for (int k = 0; k < innerGradient.length; k++) {
        innerGradient[k] = operand.get(k % operand.size()) > 0.0 ? outerFunctionGradient[k] : 0.0;
      }
  
      ((Chainable) operand).backpropagate(innerGradient);
    }
  }

  public static double[] relu(Tensor operand) {
    double[] data = new double[operand.size()];
    for (int i = 0; i < data.length; i++) {
      data[i] = Math.max(0.0, operand.get(i));
    }
    return data;
  }

}
