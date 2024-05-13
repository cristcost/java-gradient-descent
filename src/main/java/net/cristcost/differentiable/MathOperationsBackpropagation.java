package net.cristcost.differentiable;

class MathOperationsBackpropagation {

  static void sum(double outerFunctionGradient, Scalar... operands) {
    for (Scalar operand : operands) {
      if (operand instanceof Chainable) {
        ((Chainable) operand).backpropagate(outerFunctionGradient);
      }
    }
  }

  static void multiply(double outerFunctionGradient, Scalar... operands) {
    for (int i = 0; i < operands.length; i++) {
      if (operands[i] instanceof Chainable) {
        Chainable computed = (Chainable) operands[i];
        double innerFunctionFradient = outerFunctionGradient;
        for (int j = 0; j < operands.length; j++) {
          if (i != j)
            innerFunctionFradient *= operands[j].getValue();
        }

        computed.backpropagate(innerFunctionFradient);
      }
    }
  }

  static void pow(double outerFunctionGradient, Scalar base, Scalar exponent) {
    if (base instanceof Chainable) {
      ((Chainable) base).backpropagate(outerFunctionGradient * exponent.getValue()
          * Math.pow(base.getValue(), exponent.getValue() - 1));
    }

    if (exponent instanceof Chainable) {
      ((Chainable) exponent).backpropagate(
          outerFunctionGradient
              * Math.log(base.getValue())
              * Math.pow(base.getValue(), exponent.getValue()));
    }
  }

  static void relu(double outerFunctionGradient, Scalar operand) {
    if (operand instanceof Chainable) {
      ((Chainable) operand).backpropagate(operand.getValue() > 0.0 ? outerFunctionGradient : 0.0);
    }
  }
}
