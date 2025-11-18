from tf_agents.networks.network import DistributionNetwork
from tf_agents.distributions.gumbel_softmax import GumbelSoftmax

from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec
import tensorflow as tf
import tensorflow_probability as tfp

class GumbelProjectionNetwork(DistributionNetwork):

    """Generates a tfp.distribution.Categorical using Gumbel-Softmax."""

    def __init__(
        self,
        sample_spec,
        logits_init_output_factor=0.1,
        name='GumbelProjectionNetwork',
    ):
        """Creates an instance of GumbelProjectionNetwork.

        Args:
          sample_spec: A `tensor_spec.BoundedTensorSpec` detailing the shape and
            dtypes of samples pulled from the output distribution.
          logits_init_output_factor: Output factor for initializing kernel logits
            weights.
          name: A string representing name of the network.
        """
        output_spec = self._output_distribution_spec(
            sample_spec.shape,
            sample_spec,
            name,
        )

        super(GumbelProjectionNetwork, self).__init__(
            input_tensor_spec=None,
            state_spec=(),
            output_spec=output_spec,
            name=name,
        )
        self._projection_layer = tf.keras.layers.Dense(
            tf.reduce_prod(output_spec.input_params_spec.shape),
            activation=None,
        )

    def _output_distribution_spec(self, output_shape, sample_spec, name):
        input_params_spec = tensor_spec.TensorSpec(
            shape=output_shape, dtype=tf.float32, name=name + '_logits'
        )
        return distribution_spec.DistributionSpec(
            GumbelSoftmax,
            input_params_spec=input_params_spec,
            sample_spec=sample_spec,
            dtype=tf.float32,
        )
    
    def call(self, inputs, outer_rank, training=False, mask=None):
        logits = self._projection_layer(inputs)
        distribution = GumbelSoftmax(
            logits=logits, temperature=0.3, validate_args=False, dtype=tf.float32
        )
        return distribution.sample(), () 