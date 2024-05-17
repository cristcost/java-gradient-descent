package net.cristcost.differentiable;

public interface Broadcastable {

  Tensor broadcast(int... shape);
}
