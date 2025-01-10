mod matmul_warmup;
mod tests;

use std::{path::PathBuf, time::Duration};

use matmul_warmup::warmup;
use tests::{decompress_jpeg, decompress_png, denoise_images, reconstruct_images};

fn calculate_stats(latencies: &Vec<(PathBuf, Duration, Duration)>) {
    // calculate min, max, median and mean for load and decompress times, output in microseconds
    let mut load_times: Vec<u128> = latencies.iter().map(|x| x.1.as_micros()).collect();
    let mut dec_times: Vec<u128> = latencies.iter().map(|x| x.2.as_micros()).collect();

    load_times.sort_unstable();
    dec_times.sort_unstable();

    let stats = |data: &Vec<u128>| {
        let len = data.len();
        let min = data[0];
        let max = data[len - 1];
        let median = if len % 2 == 0 {
            (data[len / 2 - 1] + data[len / 2]) as f64 / 2.0
        } else {
            data[len / 2] as f64
        };
        let mean = data.iter().sum::<u128>() as f64 / len as f64;
        (min, max, median, mean)
    };

    let (l_min, l_max, l_median, l_mean) = stats(&load_times);
    let (d_min, d_max, d_median, d_mean) = stats(&dec_times);

    println!(
        "Load times -> min: {:.2} µs, max: {:.2} µs, median: {:.2} µs, mean: {:.2} µs",
        l_min, l_max, l_median, l_mean
    );
    println!(
        "Decompress times -> min: {:.2} µs, max: {:.2} µs, median: {:.2} µs, mean: {:.2} µs",
        d_min, d_max, d_median, d_mean
    );
}

/// Main program entry point
fn main() {
    // Warmup the CPU cache with matrix multiplication
    warmup(10);

    let folder = "./images";

    println!("--- Starting Decompression Benchmark ---");

    // Decompress PNG and measure latencies
    match decompress_png(folder) {
        Ok(latencies) => {
            println!("PNG Decompression Results:");
            calculate_stats(&latencies);
        }
        Err(err) => eprintln!("Error processing PNG images: {}", err),
    }

    // Decompress JPEG and measure latencies
    match decompress_jpeg(folder) {
        Ok(latencies) => {
            println!("\nJPEG Decompression Results:");
            calculate_stats(&latencies);
        }
        Err(err) => eprintln!("Error processing JPEG images: {}", err),
    }

    let model_recon = [
        ("./models/small.onnx", 64, 16),
        ("./models/base.onnx", 256, 32),
        ("./models/large.onnx", 1024, 64),
        ("./models/small_int8.onnx", 64, 16),
        ("./models/base_int8.onnx", 256, 32),
        ("./models/large_int8.onnx", 1024, 64),
    ];
    let model_path_denoise = "./models/dncnn.onnx";

    // convert to cwd + model_path
    let model_path_recon: Vec<String> = model_recon
        .iter()
        .map(|x| {
            format!(
                "{}/{}",
                std::env::current_dir().unwrap().to_str().unwrap(),
                x.0
            )
        })
        .collect();
    let model_path_denoise = format!(
        "{}/{}",
        std::env::current_dir().unwrap().to_str().unwrap(),
        model_path_denoise
    );

    for i in 0..model_path_recon.len() {
        // Reconstruct images using the ONNX decoder model
        match reconstruct_images(&model_path_recon[i], model_recon[i].1, model_recon[i].2) {
            Ok(latencies) => {
                println!("\nONNX Model Inference Results: {}", model_path_recon[i]);
                calculate_stats(&latencies);
            }
            Err(err) => eprintln!("Error processing ONNX model: {}", err),
        }
    }

    // Denoise images using the ONNX denoiser model
    match denoise_images(&model_path_denoise) {
        Ok(latencies) => {
            println!("\nONNX Denoiser Inference Results:");
            calculate_stats(&latencies);
        }
        Err(err) => eprintln!("Error processing ONNX denoiser model: {}", err),
    }
}
