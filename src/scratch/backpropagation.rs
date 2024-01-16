use ndarray::array;
use ndarray_rand::rand_distr::num_traits::Float;

pub(crate) fn simple_backpropagation() {
    let x = array![1., -2., 3.];
    let w = array![-3., -1., 2.];
    let b = 1.;

    let xw0 = x[0] * w[0];
    let xw1 = x[1] * w[1];
    let xw2 = x[2] * w[2];

    println!("{}, {}, {}", xw0, xw1, xw2);

    let z = xw0 + xw1 + xw2 + b;
    // DEBUG: println! z
    println!("z >>> {:#?}", z);

    let y = z.max(0.);
    // DEBUG: println! y
    println!("y >>> {:#?}", y);
}
