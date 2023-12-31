use ndarray::{
    s,
    Array,
    Array1,
    Array2,
    ArrayD,
    Axis,
    Ix2,
};
use ndarray_rand::{
    rand::{
        Rng,
        SeedableRng,
    },
    rand_distr::{
        Distribution,
        Uniform,
    },
    RandomExt,
};
use rand_isaac::Isaac64Rng;

#[derive(Default, Debug)]
struct Model {
    inputs: Array2<f64>,
    output: Array2<f64>,
}

impl Model {
    fn load_inputs(&mut self, inputs: Array2<f64>) {
        self.inputs = inputs
    }
}

trait DenseLayer {
    fn dense_forward(&mut self, weights: Array2<f64>, biases: Array2<f64>);
}

impl DenseLayer for Model {
    fn dense_forward(&mut self, weights: Array2<f64>, biases: Array2<f64>) {
        self.output = self.inputs.dot(&weights) + &biases;
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

    fn categorical_cross_entropy_forward(
        &self,
        y_true: ArrayD<usize>,
    ) -> Array1<f64>;
}

impl Loss for Model {
    fn calculate_mean_loss(&self, sample_losses: &Array1<f64>) -> f64 {
        sample_losses.mean().unwrap()
    }

    fn categorical_cross_entropy_forward(
        &self,
        y_true: ArrayD<usize>,
    ) -> Array1<f64> {
        let y_pred = &self.output;
        // clip data to prevent division by 0
        // clip both sides to not drag mean towards any value
        let y_pred_clipped = y_pred.mapv(|v| v.max(1e-7).min(1.0 - 1e-7));

        let shape_len = y_true.ndim();
        let correct_confidences = match shape_len {
            // probabilities for target values
            // only if categorical labels
            1 => y_true
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

pub(crate) fn run() {
    let n_samples = 100;
    let n_features = 3;

    let mut rng = Isaac64Rng::seed_from_u64(1);
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

    let mut model = Model::default();
    model.load_inputs(inputs);
    println!("model inputs >>> {:#?}", model.inputs.slice(s![0..5, ..]));

    let weights1 =
        Array::random_using((n_features, 2), Uniform::new(-1.0, 1.0), &mut rng)
            * 0.01;
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

    let weights2 =
        Array::random_using((3, 3), Uniform::new(-1.0, 1.0), &mut rng) * 0.01;
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

    let sample_losses =
        model.categorical_cross_entropy_forward(y_true.into_dyn());
    let loss = model.calculate_mean_loss(&sample_losses);

    // DEBUG: println! loss
    println!("loss >>> {:#?}", loss);
}
