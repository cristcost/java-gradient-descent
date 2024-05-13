package net.cristcost.differentiable;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class ConstantScalar implements Scalar {

  @Getter
  private final double value;

}
