use ndarray::{
    array,
    Array1,
    Array2,
    Axis,
};
use ndarray_rand::rand_distr::num_traits::Float;

pub(crate) fn categorical_cross_entropy_loss() {
    let softmax_outputs =
        array![[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08],];

    // clip data to prevent division by 0
    // clip both sides to not drag mean towards
    // any value
    let softmax_outputs = softmax_outputs.mapv(|v| v.max(1e-7).min(1.0 - 1e-7));

    let class_targets = array![[1, 0, 0], [0, 1, 0], [0, 1, 0]];
    let shape_len = class_targets.shape().len();
    let correct_confidences = match shape_len {
        // probabilities for target values
        // only if categorical labels
        1 => class_targets
            .iter()
            .enumerate()
            .map(|(i, &target)| -softmax_outputs[(i, target)])
            .collect::<Array1<f64>>(),
        // Mask values
        // only for one-hot encoded labels
        2 => (softmax_outputs * class_targets.mapv(|v| v as f64))
            .sum_axis(Axis(1)),
        _ => panic!("OOPS! Unexpected lable types."),
    };
    // DEBUG: println! correct_confidences
    println!("correct_confidences >>> {:#?}", correct_confidences);

    let neg_log = -correct_confidences.mapv(f64::ln);

    let average_loss = neg_log.mean().unwrap();

    println!("Average loss: {}", average_loss);
}
