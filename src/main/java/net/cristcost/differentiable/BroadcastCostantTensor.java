package net.cristcost.differentiable;

import java.util.Arrays;

public class BroadcastCostantTensor implements Tensor {

  private final Tensor originalTensor;
  private final int[] broadcastShape;

  public BroadcastCostantTensor(Tensor originalTensor, int[] broadcastShape) {
    if (originalTensor.getShape().length != 0) {
      throw new UnsupportedOperationException("Broadcast of non scalar values not supported");
    }
    this.originalTensor = originalTensor;
    this.broadcastShape = broadcastShape;
  }

  @Override
  public int size() {
    return Arrays.stream(broadcastShape).reduce(1, (a, b) -> a * b);
  }

  @Override
  public double get(int... indices) {
    if (originalTensor.getShape().length == 0) {
      return getData()[0];
    } else {
      throw new UnsupportedOperationException("Broadcast of non scalar values not supported");
    }
  }


  @Override
  public double[] getData() {
    return originalTensor.getData();
  }

  @Override
  public int[] getShape() {
    return broadcastShape;
  }

}
