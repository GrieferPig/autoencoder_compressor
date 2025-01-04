use ndarray::array;
use ndarray::{Array, Array2};
use rand::distributions::Uniform;
use rand::prelude::*;
use std::time::Instant;

// Function to generate a square matrix filled with random floating-point numbers
fn generate_matrix(size: usize) -> Array2<f64> {
    let mut rng = thread_rng();
    let uniform = Uniform::new(0.0, 1.0);
    Array::from_shape_fn((size, size), |_| rng.sample(&uniform))
}

// Function to perform matrix multiplication using ndarray's dot method
fn multiply_matrices(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    a.dot(b)
}

pub fn warmup(rounds: usize) {
    for _ in 0..rounds {
        // Define the size of the matrices (adjust based on your cache size)
        let size = 1000; // For example, 1000x1000 matrices

        println!("Generating matrices...");
        let a = generate_matrix(size);
        let b = generate_matrix(size);

        println!("Starting cache-intensive operation (Matrix Multiplication)...");
        let start = Instant::now();
        let c = multiply_matrices(&a, &b);
        let duration = start.elapsed();

        println!(
            "Matrix multiplication completed in {:.2?} seconds.",
            duration
        );

        // Optionally, to prevent the compiler from optimizing away the computation,
        // you can print a checksum of the result matrix.
        let checksum: f64 = c.sum();
        println!("Checksum of result matrix: {}", checksum);
    }
}
