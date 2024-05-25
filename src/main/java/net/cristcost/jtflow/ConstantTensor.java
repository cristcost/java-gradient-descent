package net.cristcost.jtflow;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class ConstantTensor implements Tensor, Broadcastable {

  @Getter
  private final double[] data;

  @Getter
  private final int[] shape;

  @Override
  public String toString() {
    return this.json();
  }
}
