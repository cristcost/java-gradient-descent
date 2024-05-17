package net.cristcost.differentiable;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class ConstantTensor implements Tensor {

  @Getter
  private final NDimensionalArray value;

}
