package net.cristcost.differentiable;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class VariableScalar implements Scalar, Differentiable {
  @Getter
  private final double value;

  @Getter
  private double gradient = 0.0;

  @Override
  public void backpropagate(double outerGradient) {
    this.gradient += outerGradient;
  }

}
