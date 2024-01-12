use ndarray::{
    s,
    Array,
    Array1,
    Array2,
    ArrayD,
    Axis,
    Ix1,
    Ix2,
};
use ndarray_rand::{
    rand::{
        Rng,
        SeedableRng,
    },
    rand_distr::{
        Distribution,
        Normal,
        Uniform,
    },
    RandomExt,
};
use rand_isaac::Isaac64Rng;

#[derive(Default, Debug)]
struct Model {
    inputs: Array2<f64>,
    y_true: ArrayD<usize>,
    output: Array2<f64>,
}

impl Model {
    fn default(n_features: usize) -> Model {
        Model {
            ..Default::default()
        }
    }

    fn load_inputs(&mut self, inputs: Array2<f64>) {
        self.inputs = inputs
    }

    fn load_y_true(&mut self, y_true: ArrayD<usize>) {
        self.y_true = y_true
    }
}

trait DenseLayer {
    fn dense_forward(&mut self, weights: &Array2<f64>, biases: &Array2<f64>);
}

impl DenseLayer for Model {
    fn dense_forward(&mut self, weights: &Array2<f64>, biases: &Array2<f64>) {
        self.output = self.inputs.dot(weights) + biases;
    }
}

trait ActivationReLU {
    fn relu_forward(&mut self);
}

impl ActivationReLU for Model {
    fn relu_forward(&mut self) {
        self.output = self.output.mapv(|a: f64| a.max(0.0));
    }
}

trait ActivationSoftmax {
    fn softmax_forward(&mut self);
}

impl ActivationSoftmax for Model {
    fn softmax_forward(&mut self) {
        let exp_values = self.output.mapv(|v: f64| v.exp());

        let n_dim = self.output.ndim();
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

trait Loss {
    fn calculate_mean_loss(&self, sample_losses: &Array1<f64>) -> f64;

    fn categorical_cross_entropy_forward(&self) -> Array1<f64>;
}

impl Loss for Model {
    fn calculate_mean_loss(&self, sample_losses: &Array1<f64>) -> f64 {
        sample_losses.mean().unwrap()
    }

    fn categorical_cross_entropy_forward(&self) -> Array1<f64> {
        let y_true = self.y_true.clone();
        let y_pred = &self.output;
        // clip data to prevent division by 0
        // clip both sides to not drag mean towards any value
        let y_pred_clipped = y_pred.mapv(|v| v.max(1e-7).min(1.0 - 1e-7));

        let shape_len = self.y_true.ndim();
        let correct_confidences = match shape_len {
            // probabilities for target values
            // only if categorical labels
            1 => self
                .y_true
                .iter()
                .enumerate()
                .map(|(i, &target)| -y_pred_clipped[(i, target)])
                .collect::<Array1<f64>>(),
            // Mask values
            // only for one-hot encoded labels
            2 => (y_pred_clipped
                * y_true
                    .into_dimensionality::<Ix2>()
                    .unwrap()
                    .mapv(|v| v as f64))
            .sum_axis(Axis(1)),
            _ => panic!("OOPS! Unexpected lable types."),
        };

        -correct_confidences.mapv(f64::ln)
    }
}

trait Accuracy {
    fn calculate_accuracy(&mut self) -> f64;
}

impl Accuracy for Model {
    fn calculate_accuracy(&mut self) -> f64 {
        let y_true = self.y_true.clone();

        let predictions = &mut self.output;
        let pred_argmax = predictions.map_axis_mut(Axis(1), |subview| {
            subview
                .indexed_iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        });

        let shape_len = self.y_true.ndim();
        let y_true = match shape_len {
            1 => y_true.into_dimensionality::<Ix1>().unwrap(),
            2 => y_true.into_dimensionality::<Ix2>().unwrap().map_axis_mut(
                Axis(1),
                |subview| {
                    subview
                        .indexed_iter()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap_or(0)
                },
            ),
            _ => panic!("OOPS! Unexpected lable types."),
        };

        let correct_count = pred_argmax
            .iter()
            .zip(self.y_true.iter())
            .filter(|(&p, &t)| p == t)
            .count();
        correct_count as f64 / pred_argmax.len() as f64
    }
}

pub(crate) fn train_model() {
    let n_samples = 100;
    let n_features = 3;

    let mut rng = Isaac64Rng::seed_from_u64(8);
    let inputs: Array2<f64> = Array::random_using(
        (n_samples, n_features),
        Uniform::new(0.0, 1.0),
        &mut rng,
    );

    // Generate one-hot encoded labels
    let mut y_true = Array2::<usize>::zeros((n_samples, n_features));
    for (i, mut row) in y_true.outer_iter_mut().enumerate() {
        let class_idx = rng.gen_range(0..n_features);
        row[class_idx] = 1;
    }

    let mut model = Model::default(n_features);
    model.load_inputs(inputs);
    // DEBUG: println! inputs
    println!("model inputs >>> {:#?}", model.inputs.slice(s![0..5, ..]));

    model.load_y_true(y_true.into_dyn());
    // DEBUG: println! y_true
    println!("model y_true >>> {:#?}", model.y_true.slice(s![0..5, ..]));

    let mut dense1_weights =
        Array::random_using((n_features, 2), Uniform::new(-1.0, 1.0), &mut rng)
            * 0.01;
    let mut dense1_biases: Array2<f64> = Array::zeros((1, 2));
    let mut dense2_weights =
        Array::random_using((3, 3), Uniform::new(-1.0, 1.0), &mut rng) * 0.01;
    let mut dense2_biases: Array2<f64> = Array::zeros((1, 3));

    let mut lowest_loss = 999999.;
    let mut best_dense1_weights = dense1_weights.clone();
    let mut best_dense1_biases = dense1_biases.clone();
    let mut best_dense2_weights = dense2_weights.clone();
    let mut best_dense2_biases = dense2_biases.clone();

    const ITERATIONS: usize = 100;

    for idx in 1..ITERATIONS {
        let dense1_weights_clone = dense1_weights.clone();
        let dense1_biases_clone = dense1_biases.clone();
        let dense2_weights_clone = dense2_weights.clone();
        let dense2_biases_clone = dense2_biases.clone();

        update_weights_and_biases(
            n_features,
            &mut dense1_weights,
            &mut dense1_biases,
            &mut dense2_weights,
            &mut dense2_biases,
        );

        let (loss, accuracy) = training_iteration(
            &mut model,
            n_features,
            &dense1_weights,
            &dense1_biases,
            &dense2_weights,
            &dense2_biases,
        );
        println!("Iter: {}, loss: {}, acc: {}", idx, loss, accuracy);

        // NEED FIX??
        // The weights and biases freeze at the values of previous best results
        // if the revertion happens once
        if loss < lowest_loss {
            println!(
                "New set of weights found, iteration: {}, loss: {}, acc: {}",
                idx, loss, accuracy
            );
            best_dense1_weights = dense1_weights_clone;
            best_dense1_biases = dense1_biases_clone;
            best_dense2_weights = dense2_weights_clone;
            best_dense2_biases = dense2_biases_clone;
            lowest_loss = loss
        } else {
            dense1_weights = best_dense1_weights.clone();
            dense1_biases = best_dense1_biases.clone();
            dense2_weights = best_dense2_weights.clone();
            dense2_biases = best_dense2_biases.clone();
        }
    }
}

fn training_iteration(
    model: &mut Model,
    n_features: usize,
    best_dense1_weights: &Array2<f64>,
    best_dense1_biases: &Array2<f64>,
    best_dense2_weights: &Array2<f64>,
    best_dense2_biases: &Array2<f64>,
) -> (f64, f64) {
    model.dense_forward(best_dense1_weights, best_dense1_biases);
    // println!(
    //     "dense layer 1 output >>> {:#?}",
    //     model.output.slice(s![0..5, ..])
    // );
    model.relu_forward();
    // println!(
    //     "relu forward output >>> {:#?}",
    //     model.output.slice(s![0..5, ..])
    // );

    model.dense_forward(best_dense2_weights, best_dense2_biases);
    // println!(
    //     "dense layer 2 output >>> {:#?}",
    //     model.output.slice(s![0..5, ..])
    // );
    model.softmax_forward();
    // println!(
    //     "softmax forward output >>> {:#?}",
    //     model.output.slice(s![0..5, ..])
    // );

    let accuracy = model.calculate_accuracy();
    // // DEBUG: println! accuracy
    // println!("accuracy >>> {:#?}", accuracy);

    let sample_losses = model.categorical_cross_entropy_forward();
    let loss = model.calculate_mean_loss(&sample_losses);

    // // DEBUG: println! loss
    // println!("loss >>> {:#?}", loss);

    (loss, accuracy)
}

fn update_weights_and_biases(
    n_features: usize,
    dense1_weights: &mut Array2<f64>,
    dense1_biases: &mut Array2<f64>,
    dense2_weights: &mut Array2<f64>,
    dense2_biases: &mut Array2<f64>,
) {
    let mut rng = Isaac64Rng::seed_from_u64(0);
    let normal_dist = Normal::new(0.0, 1.0).unwrap();

    *dense1_weights +=
        &(Array2::random_using((n_features, 2), normal_dist, &mut rng) * 0.05);
    *dense1_biases +=
        &(Array2::random_using((1, 2), normal_dist, &mut rng) * 0.05);
    *dense2_weights +=
        &(Array2::random_using((3, 3), normal_dist, &mut rng) * 0.05);
    *dense2_biases +=
        &(Array2::random_using((1, 3), normal_dist, &mut rng) * 0.05);
}
