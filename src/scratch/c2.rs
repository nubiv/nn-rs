pub(crate) fn run_nn() {
    let inputs = [1.0, 2.0, 3.0, 2.5];

    let weights1 = [0.2, 0.8, -0.5, 1.0];
    let bias1 = 2.0;
    let weights2 = [0.5, -0.91, 0.26, -0.5];
    let bias2 = 3.0;
    let weights3 = [-0.26, -0.27, 0.17, 0.87];
    let bias3 = 0.5;

    let mut weights: Vec<[f64; 4]> = vec![];
    weights.push(weights1);
    weights.push(weights2);
    weights.push(weights3);
    let mut bias: Vec<f64> = vec![];
    bias.push(bias1);
    bias.push(bias2);
    bias.push(bias3);

    let mut outputs: Vec<f64> = vec![];

    for (neuron_weights, neuron_bias) in weights.iter().zip(&bias) {
        single_neuron_proc(&inputs, neuron_weights, neuron_bias, &mut outputs);
    };

    println!("outputs >>> {:#?} ", outputs);

    // single_neuron();
}

fn single_neuron() {
    let inputs = vec![1.0, 2.0, 3.0, 2.5];
    let weights = vec![0.2, 0.8, -0.5, 1.0];
    let bias = 2.0;

    let output = inputs[0]  * weights[0] + inputs[1]  * weights[1] + inputs[2]  * weights[2] + inputs[3] * weights[3] + bias ;  
    println!("output >>> {}", output);
}

fn single_neuron_proc(inputs: &[f64], weights: &[f64], bias: &f64, outputs: &mut Vec<f64>) {
    let output = inputs[0]  * weights[0] + inputs[1]  * weights[1] + inputs[2]  * weights[2] + inputs[3] * weights[3] + bias;   

    outputs.push(output);
}