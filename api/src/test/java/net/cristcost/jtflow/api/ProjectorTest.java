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

    assertTrue(differentiableProjector.differentiable(d -> d.getGradient()).isPresent());
    assertTrue(differentiableProjector.chainable(ProjectorTest::dummyChainOperation).isPresent());
    assertFalse(differentiableProjector.broadcastable(b -> b.broadcast(1, 2, 3)).isPresent());
  }

  @Test
  void testBroadcastableProjector() {

    Tensor broadcastableProjector = new BroadcastableProjector();

    assertFalse(broadcastableProjector.differentiable(d -> d.getGradient()).isPresent());
    assertFalse(broadcastableProjector.chainable(ProjectorTest::dummyChainOperation).isPresent());
    assertTrue(broadcastableProjector.broadcastable(b -> b.broadcast(1, 2, 3)).isPresent());
  }

  @Test
  void testChainableProjector() {

    Tensor chainableProjector = new ChainableProjector();

    assertFalse(chainableProjector.differentiable(d -> d.getGradient()).isPresent());
    assertTrue(chainableProjector.chainable(ProjectorTest::dummyChainOperation).isPresent());
    assertFalse(chainableProjector.broadcastable(b -> b.broadcast(1, 2, 3)).isPresent());
  }

  private static Boolean dummyChainOperation(Chainable c) {
    c.backpropagate(null);
    return true;
  }
}
