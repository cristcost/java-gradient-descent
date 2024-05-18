package net.cristcost.verify;

import static net.cristcost.differentiable.MathLibrary.*;
import net.cristcost.differentiable.ComputationGraphStats;
import net.cristcost.differentiable.Tensor;

public class CheckSampleComputationalGraphs {

  public static Tensor fibonacci(int n) {
    if (n > 1) {
      return sum(fibonacci(n - 1), fibonacci(n - 2));
    } else {
      return scalar(n);
    }
  }

  public static Tensor factorial(int n) {
    if (n > 1) {
      return multiply(scalar(n), factorial(n - 1));
    } else {
      return scalar(1);
    }
  }

  public static Tensor reluQuadraticFunction(double value) {
    Tensor x = scalar(value);

    return relu(sum(
        multiply((scalar(-0.5)), pow(x, scalar(2.0))),
        multiply(scalar(2.0), x),
        scalar(6.0)));
  }

  public static void main(String[] args) {
    System.out.println("### factorial: `10!`");
    Tensor factorial = factorial(10);
    ComputationGraphStats.printComputationGraphStats(factorial);
    System.out.println("## computation graph:");
    ComputationGraphStats.printComputationGraph(factorial);
    System.out.println();

    System.out.println("### fibonacci: `fib(7)`");
    Tensor fibonacci = fibonacci(7);
    ComputationGraphStats.printComputationGraphStats(fibonacci);
    System.out.println("## computation graph:");
    ComputationGraphStats.printComputationGraph(fibonacci);
    System.out.println();

    System.out.println("### rectified quadratic: `relu(0.5x^2 - 2x -6)` in x=2.0");
    Tensor reluQuadraticFunction = reluQuadraticFunction(2.0);
    ComputationGraphStats.printComputationGraphStats(reluQuadraticFunction);
    System.out.println("## computation graph:");
    ComputationGraphStats.printComputationGraph(reluQuadraticFunction);
    System.out.println();
  }

}
