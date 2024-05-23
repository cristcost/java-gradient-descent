package net.cristcost.mnist;

import net.cristcost.differentiable.Tensor;


interface Sample {

  int getLabel();

  Tensor getTensor();
}
