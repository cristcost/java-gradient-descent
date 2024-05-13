package net.cristcost.sgd;

import static net.cristcost.differentiable.MathLibrary.*;
import java.util.ArrayList;
import java.util.List;
import net.cristcost.differentiable.ComputationGraphStats;
import net.cristcost.differentiable.ComputedScalar;
import net.cristcost.differentiable.Scalar;
import net.cristcost.differentiable.VariableScalar;

public class MiniBatchStochasticGradientDescent {

  private static final double LEARNING_RATE = 0.1;
  private static final double TARGET_A_VALUE = 3.0;
  private static final double TARGET_B_VALUE = 1.0;

  public static void main(String[] args) {

    // y = a * x + b
    double estimatedA = Math.random();
    double estimatedB = Math.random();

    int batchSize = 5;

    for (int epoch = 0; epoch < 1000; epoch++) {
      System.out.println("=== Round " + epoch + "===");

      VariableScalar a = variable(estimatedA);
      VariableScalar b = variable(estimatedB);

      System.out.println(String.format("   a %f, b %f", a.getValue(), b.getValue()));

      List<ComputedScalar> partialLosses = new ArrayList<>();
      for (int i = 0; i < batchSize; i++) {

        double x =  -1.0 + 2.0 * Math.random();
        double y_target = TARGET_A_VALUE * x - TARGET_B_VALUE;

        ComputedScalar y_pred = sum(multiply(a, constant(x)), b);

        ComputedScalar partialMseLoss = pow(sum(y_pred, constant(-y_target)), constant(2));

        partialLosses.add(partialMseLoss);
      }

      // Loss is the average of all the partial losses
      ComputedScalar mseLoss =
          multiply(sum(partialLosses.toArray(Scalar[]::new)), constant(1.0 / batchSize));


      System.out.println(String.format("   L2 Loss %f", mseLoss.getValue()));

      if (mseLoss.getValue() < 0.0000000001) {
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
