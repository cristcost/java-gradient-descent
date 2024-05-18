package net.cristcost.differentiable;

class MathOperationsBackpropagation {

  static void sum(double[] outerFunctionGradient, Tensor... operands) {
    for (Tensor operand : operands) {
      if (operand instanceof Chainable) {
        ((Chainable) operand).backpropagate(outerFunctionGradient);
      }
    }
  }

  static void multiply(double[] outerFunctionGradient, Tensor... operands) {
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

  static void pow(double[] outerFunctionGradient, Tensor base, Tensor exponent) {
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

  static void relu(double[] outerFunctionGradient, Tensor operand) {
    if (operand instanceof Chainable) {
      double[] innerGradient = new double[outerFunctionGradient.length];
      for (int k = 0; k < innerGradient.length; k++) {
        innerGradient[k] = operand.get(k % operand.size()) > 0.0 ? outerFunctionGradient[k] : 0.0;
      }

      ((Chainable) operand).backpropagate(innerGradient);
    }
  }
}
