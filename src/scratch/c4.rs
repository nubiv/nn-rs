use ndarray::{s, Array, Array2, ArrayD, IxDyn};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use rand_isaac::Isaac64Rng;

use crate::scratch::c3::LayerDense;

struct ActivationReLU {
    output: ArrayD<f64>,
}

impl ActivationReLU {
    fn forward(&mut self, inputs: &Array2<f64>) {
        self.output = inputs.mapv(|a: f64| a.max(0.0)).into_dyn();
    }
}

pub(crate) fn activation_forward() {
    let n_samples = 100;
    let n_features = 3;

    let mut rng = Isaac64Rng::seed_from_u64(0);
    let x: Array2<f64> =
        Array::random_using((n_samples, n_features), Uniform::new(-1.0, 1.0), &mut rng);

    // Create Dense layer with 2 input features and 3 output values
    let mut dense1 = LayerDense::new(n_features, 3);

    // Perform a forward pass of our training data through this layer
    dense1.forward(&x);

    let mut activation = ActivationReLU {
        output: ArrayD::default(IxDyn::default()),
    };
    activation.forward(&dense1.output);

    println!("{:?}", activation.output.slice(s![0..5, ..]));
}
