import os
from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.keras.optimizer_v2 import adam


class AdamDenseUpdateTest(test.TestCase, parameterized.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_single_dense_update(self):
    var = variables.Variable([1.0], dtype=dtypes.float32)
    grad = constant_op.constant([1.0], dtype=dtypes.float32)
    opt = adam.NonFusedAdam(learning_rate=0.1, beta_1=0.0, beta_2=0.0, epsilon=0.1)
    self.evaluate(variables.global_variables_initializer())
    opt.apply_gradients([(grad, var)])
    expected = 1.0 - 0.1 / (1.0 + 0.1)
    self.assertAllClose(self.evaluate(var)[0], expected)


if __name__ == "__main__":
  test.main()
