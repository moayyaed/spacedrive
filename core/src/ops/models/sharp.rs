//! # SHARP Executable and Model Management
//!
//! Downloads and manages SHARP standalone executables and model weights.
//! Executables are distributed via ml-sharp-dist releases to avoid Python dependency hell.

use super::types::{ModelInfo, ModelProvider, ModelType};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

const SHARP_DIST_REPO: &str =
	"https://github.com/spacedriveapp/ml-sharp-dist/releases/latest/download";
const SHARP_MODEL_URL: &str = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharpExecutable {
	MacOSArm64,
	MacOSX64,
	WindowsX64,
	LinuxX64,
}

impl SharpExecutable {
	/// Detect platform-specific executable variant
	pub fn current_platform() -> Option<Self> {
		#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
		return Some(Self::MacOSArm64);

		#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
		return Some(Self::MacOSX64);

		#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
		return Some(Self::WindowsX64);

		#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
		return Some(Self::LinuxX64);

		#[allow(unreachable_code)]
		None
	}

	pub fn filename(&self) -> &'static str {
		match self {
			Self::MacOSArm64 | Self::MacOSX64 | Self::LinuxX64 => "sharp",
			Self::WindowsX64 => "sharp.exe",
		}
	}

	pub fn archive_name(&self) -> &'static str {
		match self {
			Self::MacOSArm64 => "sharp-macos-arm64.tar.gz",
			Self::MacOSX64 => "sharp-macos-x64.tar.gz",
			Self::WindowsX64 => "sharp-windows-x64.zip",
			Self::LinuxX64 => "sharp-linux-x64.tar.gz",
		}
	}

	pub fn download_url(&self) -> String {
		format!("{}/{}", SHARP_DIST_REPO, self.archive_name())
	}

	pub fn size_bytes(&self) -> u64 {
		match self {
			Self::MacOSArm64 => 700 * 1024 * 1024,
			Self::MacOSX64 => 750 * 1024 * 1024,
			Self::WindowsX64 => 800 * 1024 * 1024,
			Self::LinuxX64 => 700 * 1024 * 1024,
		}
	}

	pub fn id(&self) -> &'static str {
		match self {
			Self::MacOSArm64 => "sharp-exec-macos-arm64",
			Self::MacOSX64 => "sharp-exec-macos-x64",
			Self::WindowsX64 => "sharp-exec-windows-x64",
			Self::LinuxX64 => "sharp-exec-linux-x64",
		}
	}
}

pub struct SharpModel;

impl SharpModel {
	pub const FILENAME: &'static str = "sharp_2572gikvuh.pt";
	pub const ID: &'static str = "sharp-model";

	pub fn download_url() -> &'static str {
		SHARP_MODEL_URL
	}

	pub fn size_bytes() -> u64 {
		450 * 1024 * 1024
	}
}

/// Manages SHARP executable and model downloads
pub struct SharpManager {
	sharp_dir: PathBuf,
}

impl SharpManager {
	pub fn new(data_dir: &Path) -> Self {
		Self {
			sharp_dir: data_dir.join("models").join("sharp"),
		}
	}

	/// Check if both executable and model are downloaded
	pub async fn is_available(&self) -> bool {
		self.is_executable_downloaded().await && self.is_model_downloaded().await
	}

	/// Check if platform-specific executable exists
	pub async fn is_executable_downloaded(&self) -> bool {
		if let Some(platform) = SharpExecutable::current_platform() {
			let path = self.sharp_dir.join(platform.filename());
			path.exists()
		} else {
			false
		}
	}

	/// Check if model weights exist with size verification
	pub async fn is_model_downloaded(&self) -> bool {
		let path = self.sharp_dir.join(SharpModel::FILENAME);
		if !path.exists() {
			return false;
		}

		// Verify size with 5% tolerance
		if let Ok(metadata) = tokio::fs::metadata(&path).await {
			let expected = SharpModel::size_bytes();
			let actual = metadata.len();
			let tolerance = expected / 20;
			actual.abs_diff(expected) < tolerance
		} else {
			false
		}
	}

	pub fn get_executable_path(&self) -> Option<PathBuf> {
		SharpExecutable::current_platform().map(|p| self.sharp_dir.join(p.filename()))
	}

	pub fn get_model_path(&self) -> PathBuf {
		self.sharp_dir.join(SharpModel::FILENAME)
	}

	/// List all SHARP components (executable + model) with download status
	pub async fn list_components(&self) -> Result<Vec<ModelInfo>> {
		let mut components = Vec::new();

		// Add executable
		if let Some(platform) = SharpExecutable::current_platform() {
			components.push(ModelInfo {
				id: platform.id().to_string(),
				name: format!("SHARP Executable ({})", platform.archive_name()),
				model_type: ModelType::Sharp,
				size_bytes: platform.size_bytes(),
				provider: ModelProvider::GitHub {
					owner: "spacedriveapp".to_string(),
					repo: "ml-sharp-dist".to_string(),
				},
				filename: platform.filename().to_string(),
				downloaded: self.is_executable_downloaded().await,
				description: Some(
					"Standalone SHARP executable with bundled dependencies".to_string(),
				),
			});
		}

		// Add model
		components.push(ModelInfo {
			id: SharpModel::ID.to_string(),
			name: "SHARP Model Weights".to_string(),
			model_type: ModelType::Sharp,
			size_bytes: SharpModel::size_bytes(),
			provider: ModelProvider::Direct {
				url: SharpModel::download_url().to_string(),
			},
			filename: SharpModel::FILENAME.to_string(),
			downloaded: self.is_model_downloaded().await,
			description: Some("Pre-trained SHARP model for view synthesis".to_string()),
		});

		Ok(components)
	}
}
