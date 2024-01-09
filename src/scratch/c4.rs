use ndarray::{
    array,
    s,
    Array,
    Array1,
    Array2,
    ArrayD,
    Axis,
    IxDyn,
};
use ndarray_rand::{
    rand::SeedableRng,
    rand_distr::Uniform,
    RandomExt,
};
use rand_isaac::Isaac64Rng;

use crate::scratch::c3::LayerDense;

struct ActivationReLU {
    output: ArrayD<f64>,
}

impl ActivationReLU {
    fn forward(&mut self, inputs: &ArrayD<f64>) {
        self.output = inputs.mapv(|a: f64| a.max(0.0)).into_dyn();
    }
}

pub(crate) fn activation_forward() {
    let n_samples = 100;
    let n_features = 3;

    let mut rng = Isaac64Rng::seed_from_u64(0);
    let x: Array2<f64> = Array::random_using(
        (n_samples, n_features),
        Uniform::new(-1.0, 1.0),
        &mut rng,
    );

    // Create Dense layer with 2 input features
    // and 3 output values
    let mut dense1 = LayerDense::new(n_features, 3);

    // Perform a forward pass of our training data
    // through this layer
    dense1.forward(&x);

    let mut activation = ActivationReLU {
        output: ArrayD::default(IxDyn::default()),
    };
    activation.forward(&dense1.output.into_dyn());

    println!("{:?}", activation.output.slice(s![0..5, ..]));
}

pub(crate) fn softmax_activation() {
    const E: f64 = std::f64::consts::E;
    let layer_outputs = [4.8, 1.21, 2.385];

    let mut exp_values: Vec<f64> = vec![];
    for output in layer_outputs {
        exp_values.push(f64::powf(E, output));
    }
    println!("exe valeus >>> {:#?}", exp_values);

    let norm_base: f64 = exp_values.iter().sum();
    let mut norm_values: Vec<f64> = vec![];
    for value in exp_values {
        norm_values.push(value / norm_base);
    }
    println!("normalized exponentiated values >>> {:#?}", norm_values);
    println!(
        "sum of normalized values >>> {:#?}",
        norm_values.iter().sum::<f64>()
    );
}

struct ActivationSoftware {
    output: ArrayD<f64>,
}

impl ActivationSoftware {
    fn forward(&mut self, inputs: &ArrayD<f64>) {
        let exp_values = inputs.mapv(|v: f64| v.exp());

        let n_dim = inputs.ndim();
        match n_dim {
            2 => {
                let sum_exp_values =
                    exp_values.sum_axis(Axis(1)).insert_axis(Axis(1));
                let probabilities = exp_values / sum_exp_values;

                self.output = probabilities
            }
            _ => {
                panic!(
                    "Cannot process the output layer with {}-dimention inputs from previous layer",
                    n_dim
                )
            }
        }
    }
}

pub(crate) fn softmax_activation_ndarray() {
    let layer_outputs = array![[4.8, 1.21, 2.385]];

    let mut activation_softmax = ActivationSoftware {
        output: ArrayD::default(IxDyn::default()),
    };

    activation_softmax.forward(&layer_outputs.into_dyn());
    println!("softmax output >>> {:#?}", activation_softmax.output);

    // let exp_values = layer_outputs.mapv(|v:
    // f64| v.exp()); println!("exponentiated
    // valeus >>> {:#?}", exp_values);

    // let sum_exp_values: f64 = exp_values.sum();
    // let norm_values = exp_values /
    // sum_exp_values; println!("normalized
    // exponentiated values >>> {:#?}",
    // norm_values);

    // println!("sum of normalized values >>> {}",
    // norm_values.sum());
}
