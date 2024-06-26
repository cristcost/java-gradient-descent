package net.cristcost.sgd;

import static net.cristcost.differentiable.MathLibrary.*;
import net.cristcost.differentiable.ComputationGraphStats;
import net.cristcost.differentiable.ComputedScalar;
import net.cristcost.differentiable.VariableScalar;

public class StochasticGradientDescent {

  private static final double LEARNING_RATE = 0.1;
  private static final double TARGET_A_VALUE = 3.0;
  private static final double TARGET_B_VALUE = 1.0;

  public static void main(String[] args) {

    // y = a * x + b
    double estimatedA = Math.random();
    double estimatedB = Math.random();

    for (int epoch = 0; epoch < 1000; epoch++) {
      System.out.println("=== Round " + epoch + "===");


      VariableScalar a = variable(estimatedA);
      VariableScalar b = variable(estimatedB);

      System.out.println(String.format("   a %f, b %f", a.getValue(), b.getValue()));

      double x = -1.0 + 2.0 * Math.random();
      double y_target = TARGET_A_VALUE * x - TARGET_B_VALUE;

      ComputedScalar y_pred = sum(multiply(a, constant(x)), b);

      ComputedScalar mseLoss = pow(sum(y_pred, constant(-y_target)), constant(2));


      System.out.println(String.format("   Expected %f, Predicted %f, L2 Loss %f",
          y_target, y_pred.getValue(), mseLoss.getValue()));

      if (mseLoss.getValue() < 0.0000000001) {
        // Note: this is an oversimplified condition, loss is computed for a given x value and we
        // may have found an intersection point. In the general case, we need to measure the loss on
        // batch of samples that is representative enough of the function we want to learn.

        System.out.println();
        System.out.println("=== Converged to solution in " + epoch + " epochs ===");

        System.out.println("## loss function stats: ");
        ComputationGraphStats.printComputationGraphStats(mseLoss);
        System.out.println("## computation graph:");
        ComputationGraphStats.printComputationGraph(mseLoss);
        break;
      }

      mseLoss.startBackpropagation();

      System.out.println(String.format("   dL/da %f, dL/db %f",
          a.getGradient(), b.getGradient()));

      estimatedA -= a.getGradient() * LEARNING_RATE;
      estimatedB -= b.getGradient() * LEARNING_RATE;
    }
  }
}
