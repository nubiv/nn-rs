#![allow(unused_imports, dead_code, unused_variables)]

use model::train_model;
use scratch::{
    activation::{
        activation_forward,
        softmax_activation,
        softmax_activation_ndarray,
    },
    dense_layer::{
        add_layers,
        dense_layer,
    },
    derivative::numerical_derivative,
    loss::categorical_cross_entropy_loss,
    single_neuron::{
        proc_batch_data,
        run_nn,
        run_nn_ndarray,
    },
};

mod model;
mod scratch;

fn main() {
    // run_nn();
    // run_nn_ndarray();
    // proc_batch_data();
    // add_layers();
    // dense_layer();
    // activation_forward();
    // softmax_activation();
    // softmax_activation_ndarray();
    // categorical_cross_entropy_loss();
    // train_model()
    numerical_derivative(1.);
}
