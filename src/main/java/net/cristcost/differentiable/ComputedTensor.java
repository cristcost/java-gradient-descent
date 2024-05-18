package net.cristcost.differentiable;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class ComputedTensor implements Tensor {
  @Getter
  private final double[] data;

  @Getter
  private final int[] shape;

  @Getter(AccessLevel.PACKAGE)
  private final Computation fromComputation;

}
