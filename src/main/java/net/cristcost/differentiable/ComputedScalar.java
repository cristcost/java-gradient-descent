package net.cristcost.differentiable;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class ComputedScalar implements Scalar {
  @Getter
  private final double value;

  private final Computation fromComputation;
}
