use ndarray::{
    array,
    s,
    Array,
    Array2,
};
use ndarray_rand::{
    rand::SeedableRng,
    rand_distr::Uniform,
    RandomExt,
};
use rand_isaac::Isaac64Rng;

pub(crate) fn add_layers() {
    let feature_set =
        array![[1., 2., 3., 2.5], [2., 5., -1., 2.], [-1.5, 2.7, 3.3, -0.8]];
    let weights1 = array![
        [0.2, 0.8, -0.5, 1.],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ];
    let biases1 = array![2., 3., 0.5];
    let weights2 =
        array![[0.1, -0.14, 0.5], [-0.5, 0.12, -0.33], [-0.44, 0.73, -0.13]];
    let biases2 = array![-1., 2., -0.5];

    let layer1_outputs = feature_set.dot(&weights1.t()) + &biases1;
    let layer2_outputs = layer1_outputs.dot(&weights2.t()) + &biases2;
    println!("{:#?}", layer2_outputs);
}

pub(crate) struct LayerDense {
    weights: Array2<f64>,
    biases: Array2<f64>,
    pub(crate) output: Array2<f64>,
}

impl LayerDense {
    pub(crate) fn new(n_inputs: usize, n_neurons: usize) -> LayerDense {
        let mut rng = Isaac64Rng::seed_from_u64(0);

        LayerDense {
            weights: Array::random_using(
                (n_inputs, n_neurons),
                Uniform::new(0.0, 1.0),
                &mut rng,
            ) * 0.01,
            biases: Array::zeros((1, n_neurons)),
            output: Array::zeros((0, 0)),
        }
    }

    pub(crate) fn forward(&mut self, inputs: &Array2<f64>) {
        self.output = inputs.dot(&self.weights) + &self.biases;
    }
}

pub(crate) fn dense_layer() {
    let n_samples = 100;
    let n_features = 3;

    let mut rng = Isaac64Rng::seed_from_u64(0);
    let x: Array2<f64> = Array::random_using(
        (n_samples, n_features),
        Uniform::new(0.0, 1.0),
        &mut rng,
    );

    // Create Dense layer with 2 input features
    // and 3 output values
    let mut dense1 = LayerDense::new(n_features, 3);

    // Perform a forward pass of our training data
    // through this layer
    dense1.forward(&x);

    println!("{:?}", dense1.output.slice(s![0..5, ..]));
}
