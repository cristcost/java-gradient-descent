package net.cristcost.jtflow.api;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class ProjectorTest {

  private static class BroadcastableProjector extends TestTensor implements Broadcastable {
  }

  private static class ChainableProjector extends TestTensor implements Chainable {

    @Override
    public void backpropagate(double[] outerGradient) {}
  }
  private static class DifferentiableProjector extends ChainableProjector
      implements Differentiable {

    @Override
    public double[] getGradient() {
      return new double[] {};
    }
  }
  private static abstract class TestTensor implements Tensor {

    @Override
    public double[] getData() {
      return new double[] {};
    }

    @Override
    public int[] getShape() {
      return new int[] {};
    }
  }

  @Test
  void testDifferentiableProjector() {

    Tensor differentiableProjector = new DifferentiableProjector();

    assertTrue(differentiableProjector.mapDifferentiable(d -> d.getGradient()).isPresent());
    assertTrue(differentiableProjector.mapChainable(ProjectorTest::dummyChainOperation).isPresent());
    assertFalse(differentiableProjector.mapBroadcastable(b -> b.broadcast(1, 2, 3)).isPresent());
  }

  @Test
  void testBroadcastableProjector() {

    Tensor broadcastableProjector = new BroadcastableProjector();

    assertFalse(broadcastableProjector.mapDifferentiable(d -> d.getGradient()).isPresent());
    assertFalse(broadcastableProjector.mapChainable(ProjectorTest::dummyChainOperation).isPresent());
    assertTrue(broadcastableProjector.mapBroadcastable(b -> b.broadcast(1, 2, 3)).isPresent());
  }

  @Test
  void testChainableProjector() {

    Tensor chainableProjector = new ChainableProjector();

    assertFalse(chainableProjector.mapDifferentiable(d -> d.getGradient()).isPresent());
    assertTrue(chainableProjector.mapChainable(ProjectorTest::dummyChainOperation).isPresent());
    assertFalse(chainableProjector.mapBroadcastable(b -> b.broadcast(1, 2, 3)).isPresent());
  }

  private static Boolean dummyChainOperation(Chainable c) {
    c.backpropagate(null);
    return true;
  }
}
