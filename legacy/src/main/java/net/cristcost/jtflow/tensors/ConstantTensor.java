package net.cristcost.jtflow.tensors;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import net.cristcost.jtflow.api.Broadcastable;
import net.cristcost.jtflow.api.Tensor;

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
