use scratch::{
    c2::{proc_batch_data, run_nn, run_nn_ndarray},
    c3::{add_layers, dense_layer},
    c4::activation_forward,
};

mod scratch;

fn main() {
    // run_nn();
    // run_nn_ndarray();
    // proc_batch_data();
    // add_layers();
    // dense_layer();
    activation_forward();
}
