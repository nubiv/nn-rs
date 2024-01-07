use ndarray::{
    s,
    Array,
    Array2,
    ArrayD,
    Axis,
};
use ndarray_rand::{
    rand::SeedableRng,
    rand_distr::Uniform,
    RandomExt,
};
use rand_isaac::Isaac64Rng;

// struct LayerDense {
//     weights: Array2<f64>,
//     biases: Array2<f64>,
//     pub(crate) output: Array2<f64>,
// }

// impl LayerDense {
//     fn new(n_inputs: usize, n_neurons: usize)
// -> LayerDense {         let mut rng =
// Isaac64Rng::seed_from_u64(0);

//         LayerDense {
//             weights:
// Array::random_using((n_inputs, n_neurons),
// Uniform::new(0.0, 1.0), &mut rng)
//                 * 0.01,
//             biases: Array::zeros((1,
// n_neurons)),             output:
// Array::zeros((0, 0)),         }
//     }

//     fn forward(&mut self, inputs: &Array2<f64>)
// {         self.output =
// inputs.dot(&self.weights) + &self.biases;     }
// }

// struct ActivationReLU {
//     output: ArrayD<f64>,
// }

// impl ActivationReLU {
//     fn forward(&mut self, inputs: &Array2<f64>)
// {         self.output = inputs.mapv(|a: f64|
// a.max(0.0)).into_dyn();     }
// }

// struct ActivationSoftmax {
//     output: ArrayD<f64>,
// }

// impl ActivationSoftmax {
//     fn forward(&mut self, inputs: &ArrayD<f64>)
// {         let exp_values = inputs.mapv(|v: f64|
// v.exp());

//         let n_dim = inputs.ndim();
//         match n_dim {
//             2 => {
//                 let sum_exp_values =
// exp_values.sum_axis(Axis(0)).
// insert_axis(Axis(0));                 let
// probabilities = exp_values / sum_exp_values;

//                 self.output = probabilities
//             }
//             _ => {
//                 panic!(
//                     "Cannot process the output
// layer with {}-dimention inputs from previous
// layer",                     n_dim
//                 )
//             }
//         }
//     }
// }

// pub(crate) fn run_model() {
//     let n_samples = 100;
//     let n_features = 3;

//     let mut rng = Isaac64Rng::seed_from_u64(0);
//     let x: Array2<f64> =
//         Array::random_using((n_samples,
// n_features), Uniform::new(0.0, 1.0), &mut rng);

//     // Create Dense layer with 2 input features
// and 3 output values     let mut dense1 =
// LayerDense::new(n_features, 3);

//     // Perform a forward pass of our training
// data through this layer     dense1.forward(&x);
// }

/*
Put all the implementations in the Model trait instead.

struct DenseLayer {}

trait Model {}

impl Model for DenseLayer {}
*/

#[derive(Default, Debug)]
struct Model {
    inputs: Array2<f64>,
    output: Array2<f64>,
}

impl Model {
    fn load_inputs(
        &mut self,
        inputs: Array2<f64>,
    ) {
        self.inputs = inputs
    }
}

trait DenseLayer {
    fn dense_forward(
        &mut self,
        weights: Array2<f64>,
        biases: Array2<f64>,
    ) {
    }
}

impl DenseLayer for Model {
    fn dense_forward(
        &mut self,
        weights: Array2<f64>,
        biases: Array2<f64>,
    ) {
        self.output =
            self.inputs.dot(&weights) + &biases;
    }
}

trait ActivationReLU {
    fn relu_forward(&mut self) {}
}

impl ActivationReLU for Model {
    fn relu_forward(&mut self) {
        self.output =
            self.output.mapv(|a: f64| a.max(0.0));
    }
}

trait ActivationSoftmax {
    fn softmax_forward(&mut self) {}
}

impl ActivationSoftmax for Model {
    fn softmax_forward(&mut self) {
        let exp_values =
            self.output.mapv(|v: f64| v.exp());

        let n_dim = self.output.ndim();
        match n_dim {
            2 => {
                let sum_exp_values = exp_values
                    .sum_axis(Axis(1))
                    .insert_axis(Axis(1));
                let probabilities =
                    exp_values / sum_exp_values;

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

pub(crate) fn run() {
    let n_samples = 100;
    let n_features = 3;

    let mut rng = Isaac64Rng::seed_from_u64(1);
    let inputs: Array2<f64> = Array::random_using(
        (n_samples, n_features),
        Uniform::new(0.0, 1.0),
        &mut rng,
    );

    let mut model = Model::default();
    model.load_inputs(inputs);
    println!(
        "model inputs >>> {:#?}",
        model.inputs.slice(s![0..5, ..])
    );

    let weights1 = Array::random_using(
        (n_features, 2),
        Uniform::new(-1.0, 1.0),
        &mut rng,
    ) * 0.01;
    let biases1 = Array::zeros((1, 2));
    model.dense_forward(weights1, biases1);
    println!(
        "dense layer 1 output >>> {:#?}",
        model.output.slice(s![0..5, ..])
    );
    model.relu_forward();
    println!(
        "relu forward output >>> {:#?}",
        model.output.slice(s![0..5, ..])
    );

    let weights2 = Array::random_using(
        (3, 3),
        Uniform::new(-1.0, 1.0),
        &mut rng,
    ) * 0.01;
    let biases2 = Array::zeros((1, 3));
    model.dense_forward(weights2, biases2);
    println!(
        "dense layer 2 output >>> {:#?}",
        model.output.slice(s![0..5, ..])
    );
    model.softmax_forward();
    println!(
        "softmax forward output >>> {:#?}",
        model.output.slice(s![0..5, ..])
    );
}
