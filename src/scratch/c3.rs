use ndarray::array;

pub(crate) fn add_layers() {
    let feature_set = array![[1., 2., 3., 2.5], [2., 5., -1., 2.], [-1.5, 2.7, 3.3, -0.8]];
    let weights1 = array![
        [0.2, 0.8, -0.5, 1.],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ];
    let biases1 = array![2., 3., 0.5];
    let weights2 = array![[0.1, -0.14, 0.5], [-0.5, 0.12, -0.33], [-0.44, 0.73, -0.13]];
    let biases2 = array![-1., 2., -0.5];

    let layer1_outputs = feature_set.dot(&weights1.t()) + &biases1;
    let layer2_outputs = layer1_outputs.dot(&weights2.t()) + &biases2;
    println!("{:#?}", layer2_outputs);
}
