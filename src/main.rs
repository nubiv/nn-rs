#![allow(unused_imports, dead_code, unused_variables)]

use model::run;
use scratch::{
    c2::{
        proc_batch_data,
        run_nn,
        run_nn_ndarray,
    },
    c3::{
        add_layers,
        dense_layer,
    },
    c4::{
        activation_forward,
        softmax_activation,
        softmax_activation_ndarray,
    },
    c5::categorical_cross_entropy_loss,
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
    run();
}
