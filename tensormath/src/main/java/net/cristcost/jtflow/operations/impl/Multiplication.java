package net.cristcost.jtflow.operations.impl;

import java.util.Arrays;
import net.cristcost.jtflow.api.Chainable;
import net.cristcost.jtflow.api.Tensor;

// Not tested and not to be used in this form
@Deprecated()
public class Multiplication {

  public static void chain(double[] outerFunctionGradient, Tensor... operands) {
    for (int i = 0; i < operands.length; i++) {
      if (operands[i] instanceof Chainable) {
        Chainable computed = (Chainable) operands[i];
        double[] innerGradient = outerFunctionGradient.clone();
        for (int j = 0; j < operands.length; j++) {
          if (i != j) {
            for (int k = 0; k < innerGradient.length; k++) {
              // Mod over the size to broadcast implicitly
              innerGradient[k] *= operands[j].get(k % operands[j].size());
            }
          }
        }

        computed.backpropagate(innerGradient);
      }
    }
  }

  public static double[] compute(Tensor... operands) {
    double[] data = new double[operands[0].size()];
    Arrays.fill(data, 1.0);

    for (Tensor t : operands) {
      if (data.length != t.size()) {
        throw new IllegalArgumentException("Shapes do not match.");
      }
      for (int i = 0; i < data.length; i++) {
        data[i] *= t.get(i);
      }
    }
    return data;
  }

}
