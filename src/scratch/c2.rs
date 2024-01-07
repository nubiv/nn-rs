use ndarray::{
    array,
    Array,
    Array1,
    Array2,
    Dimension,
};

pub(crate) fn proc_batch_data() {
    let sample1 = [1., 2., 3., 2.5];
    let sample2 = [2., 5., -1., 2.];
    let sample3 = [-0.5, 2.7, 3.3, -0.8];
    let feature_set: Array2<f64> =
        array![sample1, sample2, sample3];

    let weights1 = [0.2, 0.8, -0.5, 1.0];
    let weights2 = [0.5, -0.91, 0.26, -0.5];
    let weights3 = [-0.26, -0.27, 0.17, 0.87];
    let weights: Array2<f64> =
        array![weights1, weights2, weights3];

    let bias1 = 2.;
    let bias2 = 3.;
    let bias3 = 0.5;
    let biases: Array1<f64> =
        array![bias1, bias2, bias3];

    let layer_outputs =
        feature_set.dot(&weights.t()) + &biases;
    println!(
        "Layout outputs >>> {:#?}",
        layer_outputs
    );
}

pub(crate) fn run_nn_ndarray() {
    let inputs = array![1.0, 2.0, 3.0, 2.5];

    let weights1 = [0.2, 0.8, -0.5, 1.0];
    let bias1 = 2.;
    let weights2 = [0.5, -0.91, 0.26, -0.5];
    let bias2 = 3.;
    let weights3 = [-0.26, -0.27, 0.17, 0.87];
    let bias3 = 0.5;

    let weights: Array2<f64> =
        array![weights1, weights2, weights3];
    let biases: Array1<f64> =
        array![bias1, bias2, bias3];

    let layer_outputs =
        weights.dot(&inputs) + &biases;
    println!(
        "Layout outputs >>> {:#?}",
        layer_outputs
    );
}

fn single_neuron_proc_ndarray<A, D>(
    inputs: &Array<A, D>,
    weights: &Array<A, D>,
    bias: &Array<A, D>,
    outputs: &mut Array<A, D>,
) where
    A: ::std::fmt::Debug,
    D: Dimension,
{
    println!("Input >>> {:#?}", inputs);
    println!("Weights >>> {:#?}", weights);
    println!("bias >>> {:#?}", bias);
}

pub(crate) fn run_nn() {
    let inputs = [1.0, 2.0, 3.0, 2.5];

    let weights1 = [0.2, 0.8, -0.5, 1.0];
    let bias1 = 2.0;
    let weights2 = [0.5, -0.91, 0.26, -0.5];
    let bias2 = 3.0;
    let weights3 = [-0.26, -0.27, 0.17, 0.87];
    let bias3 = 0.5;

    let weights: Vec<[f64; 4]> =
        vec![weights1, weights2, weights3];
    let bias: Vec<f64> =
        vec![bias1, bias2, bias3];

    let mut outputs: Vec<f64> = vec![];

    for (neuron_weights, neuron_bias) in
        weights.iter().zip(&bias)
    {
        single_neuron_proc(
            &inputs,
            neuron_weights,
            neuron_bias,
            &mut outputs,
        );
    }

    println!("outputs >>> {:#?} ", outputs);

    // single_neuron();
}

fn single_neuron() {
    let inputs = [1.0, 2.0, 3.0, 2.5];
    let weights = [0.2, 0.8, -0.5, 1.0];
    let bias = 2.0;

    let output = inputs[0] * weights[0]
        + inputs[1] * weights[1]
        + inputs[2] * weights[2]
        + inputs[3] * weights[3]
        + bias;
    println!("output >>> {}", output);
}

fn single_neuron_proc(
    inputs: &[f64],
    weights: &[f64],
    bias: &f64,
    outputs: &mut Vec<f64>,
) {
    let output = inputs[0] * weights[0]
        + inputs[1] * weights[1]
        + inputs[2] * weights[2]
        + inputs[3] * weights[3]
        + bias;

    outputs.push(output);
}
