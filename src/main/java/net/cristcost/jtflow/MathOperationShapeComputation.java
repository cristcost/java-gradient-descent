package net.cristcost.jtflow;

class MathOperationShapeComputation {

  static int[] identity(Tensor tensor) {
    return tensor.getShape();
  }

  static int[] maxShape(Tensor... operands) {
    int[] fullShape = MathLibrary.shape();
    for (Tensor t : operands) {
      if (fullShape.length < t.getShape().length) {
        fullShape = t.getShape();
      }
    }
    return fullShape.clone();
  }


}
