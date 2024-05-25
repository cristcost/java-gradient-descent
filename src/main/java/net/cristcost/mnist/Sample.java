package net.cristcost.mnist;

import net.cristcost.jtflow.Tensor;


interface Sample {

  int getLabel();

  Tensor getTensor();
}
