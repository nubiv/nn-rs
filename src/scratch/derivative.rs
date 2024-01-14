// choosee a delta small enough to approximate the derivatives as accurately as
// possible but large enough to prevent a rounding error
const P2_DELTA: f64 = 0.00001;

fn non_linear(x: f64) -> f64 {
    2. * x.powi(2)
}

pub(crate) fn numerical_derivative(x: f64) -> f64 {
    let x1 = x;
    let x2 = x1 + P2_DELTA;

    let y1 = non_linear(x1);
    let y2 = non_linear(x2);

    let approximate_derivative = (y2 - y1) / (x2 - x1);
    // DEBUG: println! approximate_derivative
    println!("approximate_derivative >>> {:#?}", approximate_derivative);

    let b = y2 - approximate_derivative * x2;

    let tangent_line = |x: f64| {
        let y = approximate_derivative * x + b;
        println!(
            "Approximate derivateive for nonlinear fn, where x = {} is {}",
            x, y
        );

        y
    };

    tangent_line(3.);

    approximate_derivative
}
