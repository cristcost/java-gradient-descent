package net.cristcost.differentiable;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class ComputedScalar implements Scalar {
  @Getter
  private final double value;

  @Getter(AccessLevel.PACKAGE)
  private final Computation fromComputation;
}
