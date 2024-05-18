package net.cristcost.differentiable;

import java.util.Arrays;
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

  @Override
  public Tensor broadcast(int... shape) {
    return new BroadcastCostantTensor(shape);
  }


  public class BroadcastCostantTensor implements Tensor {

    private final int[] broadcastShape;
    private final int broadcastSize;

    public BroadcastCostantTensor(int[] broadcastShape) {
      this.broadcastShape = broadcastShape;
      this.broadcastSize = Arrays.stream(broadcastShape).reduce(1, (a, b) -> a * b);
    }

    @Override
    public int size() {
      return broadcastSize;
    }

    @Override
    public double[] getData() {
      return ConstantTensor.this.getData();
    }

    @Override
    public int[] getShape() {
      return broadcastShape;
    }

    @Override
    public String toString() {
      return this.json();
    }

  }


}
