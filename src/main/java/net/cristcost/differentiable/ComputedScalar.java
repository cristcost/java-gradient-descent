package net.cristcost.differentiable;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class ComputedScalar implements Scalar, Chainable {
  @Getter
  private final double value;

  @Getter(AccessLevel.PACKAGE)
  private final Computation fromComputation;

  public void startBackpropagation() {
    backpropagate(1.0);
  }

  @Override
  public void backpropagate(double outerGradient) {
  }

}
