package net.cristcost.jtflow.operations.impl;

import net.cristcost.jtflow.api.Chainable;
import net.cristcost.jtflow.api.Tensor;

public class Exponentiation {

  public static void chain(double[] outerFunctionGradient, Tensor base, Tensor exponent) {
    if (base instanceof Chainable) {
      double[] innerGradient = outerFunctionGradient.clone();
  
      for (int k = 0; k < innerGradient.length; k++) {
        // Mod over the size to broadcast implicitly
        innerGradient[k] *= exponent.get(k % exponent.size())
            * Math.pow(base.get(k % base.size()), exponent.get(k % exponent.size()) - 1);
      }
      ((Chainable) base).backpropagate(innerGradient);
    }
  
    if (exponent instanceof Chainable) {
      double[] innerGradient = outerFunctionGradient.clone();
  
      for (int k = 0; k < innerGradient.length; k++) {
        // Mod over the size to broadcast implicitly
        innerGradient[k] *= Math.log(base.get(k % base.size()))
            * Math.pow(base.get(k % base.size()), exponent.get(k % exponent.size()));
      }
      ((Chainable) exponent).backpropagate(innerGradient);
    }
  }

  public static double[] pow(Tensor base, Tensor exponent) {
    if (base.size() != exponent.size()) {
      throw new IllegalArgumentException("Shapes do not match.");
    }
  
    double[] data = new double[base.size()];
    for (int i = 0; i < data.length; i++) {
      data[i] = Math.pow(base.get(i), exponent.get(i));
    }
    return data;
  }

}
