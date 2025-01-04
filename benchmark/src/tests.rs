use image::{ImageError, ImageFormat};
use ndarray::Array2;
use ort::session::Session;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Measure decompression latency for PNG images in the specified folder.
pub fn decompress_png(folder: &str) -> Result<Vec<(PathBuf, Duration, Duration)>, ImageError> {
    let mut latencies = Vec::new();
    let entries = fs::read_dir(folder)?;

    for entry in entries {
        let path = entry?.path();
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase()
            == "png"
        {
            let start = Instant::now();

            // Load file into memory
            let file_data = fs::read(&path)?;
            let file_load_time = start.elapsed();

            // Decompress PNG
            let start = Instant::now();
            image::load_from_memory_with_format(&file_data, ImageFormat::Png)?;
            let decompress_time = start.elapsed();

            // Record latencies
            latencies.push((path, file_load_time, decompress_time));
        }
    }

    Ok(latencies)
}

/// Measure decompression latency for JPEG images in the specified folder.
pub fn decompress_jpeg(folder: &str) -> Result<Vec<(PathBuf, Duration, Duration)>, ImageError> {
    let mut latencies = Vec::new();
    let entries = fs::read_dir(folder)?;

    for entry in entries {
        let path = entry?.path();
        if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
            if ext.to_lowercase() == "jpg" || ext.to_lowercase() == "jpeg" {
                let start = Instant::now();

                // Load file into memory
                let file_data = fs::read(&path)?;
                let file_load_time = start.elapsed();

                // Decompress JPEG
                let start = Instant::now();
                image::load_from_memory_with_format(&file_data, ImageFormat::Jpeg)?;
                let decompress_time = start.elapsed();

                // Record latencies
                latencies.push((path, file_load_time, decompress_time));
            }
        }
    }

    Ok(latencies)
}

/// Function to reconstruct images using the ONNX decoder model.
pub fn reconstruct_images(
    model_path: &str,
) -> Result<Vec<(PathBuf, Duration, Duration)>, Box<dyn std::error::Error>> {
    let f16 = model_path.contains("f16");
    let mut latencies = Vec::new();

    // Measure the time to load latent space into tensor
    let start = Instant::now();
    // Load the ONNX model
    let model = Session::builder()?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    // Example latent spaces (256 vectors of dimension 16)
    let latent_spaces: Array2<f32> = Array2::zeros((256, 16));
    if f16 {}

    let load_time = start.elapsed();

    // Iterate through each latent space (256 vectors of dimension 16)
    for latent_vector in latent_spaces.axis_iter(ndarray::Axis(0)) {
        let input_tensor = latent_vector.insert_axis(ndarray::Axis(0)); // Create a 1x16 tensor

        // Perform inference to reconstruct the image
        let start = Instant::now();
        let outputs = model.run(ort::inputs!["input" => input_tensor]?)?;
        let _predictions = outputs["output"].try_extract_tensor::<f32>()?;
        let inference_time = start.elapsed();
        latencies.push((PathBuf::new(), load_time, inference_time));
    }

    Ok(latencies)
}

pub fn denoise_images(
    model_path: &str,
) -> Result<Vec<(PathBuf, Duration, Duration)>, Box<dyn std::error::Error>> {
    let mut latencies = Vec::new();

    // Measure the time to load latent space into tensor
    let start = Instant::now();
    // Load the ONNX model
    let model = Session::builder()?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;
    let load_time = start.elapsed();

    for _ in 0..256 {
        // Example image
        let input_image: Array2<f32> = Array2::zeros((256, 256));

        // Perform inference to denoise the image
        let start = Instant::now();
        let outputs = model.run(ort::inputs!["image" => input_image]?)?;
        let _predictions = outputs["output0"].try_extract_tensor::<f32>()?;
        let inference_time = start.elapsed();
        latencies.push((PathBuf::new(), load_time, inference_time));
    }

    Ok(latencies)
}
