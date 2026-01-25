//! Gaussian Splat generation system
//!
//! Generates 3D Gaussian splats from images using Apple's SHARP model.
//! Generates .ply sidecar files for photorealistic view synthesis.

pub mod action;
pub mod job;
pub mod processor;

pub use action::{GenerateSplatAction, GenerateSplatInput, GenerateSplatOutput};
pub use job::{GaussianSplatJob, GaussianSplatJobConfig};
pub use processor::GaussianSplatProcessor;

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Generate a 3D Gaussian splat from an image using SHARP
///
/// Uses the downloaded SHARP executable from the model system.
/// Requires both the executable and model weights to be downloaded via model management.
///
/// # Arguments
/// * `source_path` - Path to the input image
/// * `output_dir` - Directory where the .ply file will be generated
/// * `data_dir` - Data directory for accessing downloaded SHARP components
///
/// # Returns
/// Path to the generated .ply file
pub async fn generate_splat_from_image(
	source_path: &Path,
	output_dir: &Path,
	data_dir: &Path,
) -> Result<PathBuf> {
	use tokio::process::Command;

	// Get SHARP manager
	let sharp_manager = crate::ops::models::SharpManager::new(data_dir);

	// Check if SHARP is available
	if !sharp_manager.is_available().await {
		anyhow::bail!(
			"SHARP not available. Please download SHARP executable and model from Settings > Models"
		);
	}

	let sharp_executable = sharp_manager
		.get_executable_path()
		.context("SHARP executable not found")?;
	let model_path = sharp_manager.get_model_path();

	// Ensure output directory exists
	tokio::fs::create_dir_all(output_dir).await?;

	// Build command
	let output = Command::new(&sharp_executable)
		.arg("predict")
		.arg("-i")
		.arg(source_path)
		.arg("-o")
		.arg(output_dir)
		.arg("-c")
		.arg(&model_path)
		.output()
		.await
		.context("Failed to execute SHARP")?;

	if !output.status.success() {
		let stderr = String::from_utf8_lossy(&output.stderr);
		anyhow::bail!("SHARP failed: {}", stderr);
	}

	// The output file will be named based on input filename with .ply extension
	let ply_filename = source_path
		.file_stem()
		.context("Invalid source filename")?
		.to_str()
		.context("Non-UTF8 filename")?;

	let ply_path = output_dir.join(format!("{}.ply", ply_filename));

	if !ply_path.exists() {
		anyhow::bail!(
			"SHARP did not generate expected output file: {:?}",
			ply_path
		);
	}

	Ok(ply_path)
}

/// Check if SHARP is available (both executable and model downloaded)
pub async fn check_sharp_available(data_dir: &Path) -> Result<bool> {
	let manager = crate::ops::models::SharpManager::new(data_dir);
	Ok(manager.is_available().await)
}

/// Check if an image type is supported for splat generation
pub fn is_splat_supported(mime_type: &str) -> bool {
	// SHARP supports common image formats
	matches!(
		mime_type,
		"image/jpeg" | "image/png" | "image/webp" | "image/bmp" | "image/tiff"
	)
}
