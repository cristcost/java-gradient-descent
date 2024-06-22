package net.cristcost.jtflow.optimizer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import net.cristcost.jtflow.api.Optimizer;
import net.cristcost.jtflow.operations.raw.RawMeanSquareError;

class RootMeanSquarePropagationOptimizerTest {

  final static double[] NULL_GRADIENT = {0.0, 0.0, 0.0};
  final static double[] FORWARD_GRADIENT = {1.0, 0.0, 0.0};
  final static double[] RIGHT_GRADIENT = {0.0, 1.0, 0.0};
  final static double[] UP_GRADIENT = {0.0, 0.0, 1.0};

  @Test
  void basicTest() {

    double[] parameters = {1.0, 1.0, 1.0};
    double[] lastParameters = parameters.clone();

    Optimizer optimizer = new RootMeanSquarePropagationOptimizer(0.1, 0.9);

    optimizer.optimize(parameters, NULL_GRADIENT);
    assertArrayEquals(parameters, lastParameters);

    optimizer.optimize(parameters, FORWARD_GRADIENT);
    assertThat(parameters[0]).isLessThan(lastParameters[0]);
    assertThat(parameters[1]).isEqualTo(lastParameters[1]);
    assertThat(parameters[2]).isEqualTo(lastParameters[2]);

    lastParameters = parameters.clone();
    optimizer.optimize(parameters, RIGHT_GRADIENT);
    assertThat(parameters[0]).isEqualTo(lastParameters[0]);
    assertThat(parameters[1]).isLessThan(lastParameters[1]);
    assertThat(parameters[2]).isEqualTo(lastParameters[2]);

    lastParameters = parameters.clone();
    optimizer.optimize(parameters, UP_GRADIENT);
    assertThat(parameters[0]).isEqualTo(lastParameters[0]);
    assertThat(parameters[1]).isEqualTo(lastParameters[1]);
    assertThat(parameters[2]).isLessThan(lastParameters[2]);

  }

  @Test
  void basicConvergence() {

    double[] parameters = {1.0, 1.0, 1.0};
    double[] target = {0.0, -5.0, 10.0};
    double epsilon = 1e-6;

    Optimizer optimizer = new RootMeanSquarePropagationOptimizer(0.1, 0.9);

    // Not a real gradient, just +1 or -1 if axis bigger or smaller than target
    double[] pseudoGradient = new double[3];
    int i = 0;

    while (i <= 100 && RawMeanSquareError.compute(parameters, target) > 0.001) {
      for (int j = 0; j < parameters.length; j++) {
        if (parameters[j] > target[j] + epsilon) {
          pseudoGradient[j] = +1;
        } else if (parameters[j] < target[j] - epsilon) {
          pseudoGradient[j] = -1;
        }
      }
      optimizer.optimize(parameters, pseudoGradient);
      i++;
    }

    assertThat(i).isLessThan(100);
  }
}
