//! Model download job with progress tracking

use super::{
	sharp::{SharpExecutable, SharpModel},
	types::ModelInfo,
	whisper::WhisperModel,
};
use crate::infra::job::{prelude::*, traits::DynJob};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use specta::Type;
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info};

#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct ModelDownloadConfig {
	/// Model ID to download (e.g., "whisper-base")
	pub model_id: String,
	/// Data directory for model storage
	pub data_dir: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelDownloadState {
	phase: DownloadPhase,
	model_id: String,
	download_url: String,
	target_path: PathBuf,
	temp_path: PathBuf,
	total_bytes: u64,
	downloaded_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum DownloadPhase {
	Initializing,
	Downloading,
	Verifying,
	Complete,
}

#[derive(Serialize, Deserialize)]
pub struct ModelDownloadJob {
	config: ModelDownloadConfig,
	state: ModelDownloadState,
}

impl ModelDownloadJob {
	pub fn new(config: ModelDownloadConfig) -> Self {
		Self {
			state: ModelDownloadState {
				phase: DownloadPhase::Initializing,
				model_id: config.model_id.clone(),
				download_url: String::new(),
				target_path: PathBuf::new(),
				temp_path: PathBuf::new(),
				total_bytes: 0,
				downloaded_bytes: 0,
			},
			config,
		}
	}

	pub fn for_whisper_model(model: WhisperModel, data_dir: PathBuf) -> Self {
		Self::new(ModelDownloadConfig {
			model_id: model.id().to_string(),
			data_dir,
		})
	}

	pub fn for_sharp_executable(data_dir: PathBuf) -> Option<Self> {
		let platform = SharpExecutable::current_platform()?;
		Some(Self::new(ModelDownloadConfig {
			model_id: platform.id().to_string(),
			data_dir,
		}))
	}

	pub fn for_sharp_model(data_dir: PathBuf) -> Self {
		Self::new(ModelDownloadConfig {
			model_id: SharpModel::ID.to_string(),
			data_dir,
		})
	}
}

impl Job for ModelDownloadJob {
	const NAME: &'static str = "model_download";
	const RESUMABLE: bool = true;
	const DESCRIPTION: Option<&'static str> = Some("Download AI/ML models");
}

#[async_trait::async_trait]
impl JobHandler for ModelDownloadJob {
	type Output = ModelDownloadOutput;

	async fn run(&mut self, ctx: JobContext<'_>) -> JobResult<Self::Output> {
		match self.state.phase {
			DownloadPhase::Initializing => {
				self.initialize(&ctx).await?;
				self.state.phase = DownloadPhase::Downloading;
			}
			DownloadPhase::Downloading => {}
			DownloadPhase::Verifying => {}
			DownloadPhase::Complete => {
				return Ok(ModelDownloadOutput {
					model_id: self.state.model_id.clone(),
					path: self.state.target_path.to_string_lossy().to_string(),
					size_bytes: self.state.total_bytes,
				});
			}
		}

		// Download phase
		if matches!(self.state.phase, DownloadPhase::Downloading) {
			self.download(&ctx).await?;
			self.state.phase = DownloadPhase::Verifying;
		}

		// Verify phase
		if matches!(self.state.phase, DownloadPhase::Verifying) {
			self.verify(&ctx).await?;
			self.state.phase = DownloadPhase::Complete;
		}

		ctx.log("Model download complete");

		Ok(ModelDownloadOutput {
			model_id: self.state.model_id.clone(),
			path: self.state.target_path.to_string_lossy().to_string(),
			size_bytes: self.state.total_bytes,
		})
	}
}

impl ModelDownloadJob {
	async fn initialize(&mut self, ctx: &JobContext<'_>) -> JobResult<()> {
		ctx.log(format!(
			"Initializing download for model: {}",
			self.config.model_id
		));

		// Handle SHARP executable downloads
		if self.config.model_id.starts_with("sharp-exec-") {
			if let Some(platform) = SharpExecutable::current_platform() {
				let sharp_dir = super::get_sharp_dir(&self.config.data_dir);
				tokio::fs::create_dir_all(&sharp_dir).await?;

				self.state.download_url = platform.download_url();
				self.state.target_path = sharp_dir.join(platform.filename());
				self.state.temp_path = sharp_dir.join(format!("{}.tmp", platform.filename()));
				self.state.total_bytes = platform.size_bytes();

				ctx.log(format!(
					"Downloading SHARP executable for {} ({} MB)",
					platform.archive_name(),
					self.state.total_bytes / 1024 / 1024
				));

				return Ok(());
			} else {
				return Err(JobError::execution("Unsupported platform for SHARP".into()));
			}
		}

		// Handle SHARP model downloads
		if self.config.model_id == SharpModel::ID {
			let sharp_dir = super::get_sharp_dir(&self.config.data_dir);
			tokio::fs::create_dir_all(&sharp_dir).await?;

			self.state.download_url = SharpModel::download_url().to_string();
			self.state.target_path = sharp_dir.join(SharpModel::FILENAME);
			self.state.temp_path = self.state.target_path.with_extension("tmp");
			self.state.total_bytes = SharpModel::size_bytes();

			ctx.log(format!(
				"Downloading SHARP model ({} MB) from Apple CDN",
				self.state.total_bytes / 1024 / 1024
			));

			return Ok(());
		}

		// Handle Whisper model downloads
		if let Some(model) = WhisperModel::from_str(&self.config.model_id.replace("whisper-", "")) {
			let models_dir = super::get_whisper_models_dir(&self.config.data_dir);
			tokio::fs::create_dir_all(&models_dir).await?;

			self.state.download_url = model.download_url().to_string();
			self.state.target_path = models_dir.join(model.filename());
			self.state.temp_path = self.state.target_path.with_extension("tmp");
			self.state.total_bytes = model.size_bytes();

			ctx.log(format!(
				"Downloading {} ({} MB) from Hugging Face",
				model.display_name(),
				self.state.total_bytes / 1024 / 1024
			));

			return Ok(());
		}

		Err(JobError::execution(format!(
			"Unknown model ID: {}",
			self.config.model_id
		)))
	}

	async fn download(&mut self, ctx: &JobContext<'_>) -> JobResult<()> {
		use futures::StreamExt;

		ctx.log("Starting download...");

		// Start download
		let client = reqwest::Client::new();
		let response = client
			.get(&self.state.download_url)
			.send()
			.await
			.map_err(|e| JobError::execution(format!("Download request failed: {}", e)))?;

		if !response.status().is_success() {
			return Err(JobError::execution(format!(
				"Download failed with status: {}",
				response.status()
			)));
		}

		// Verify content length
		if let Some(content_length) = response.content_length() {
			self.state.total_bytes = content_length;
		}

		// Create temp file
		let mut file = tokio::fs::File::create(&self.state.temp_path)
			.await
			.map_err(|e| JobError::execution(format!("Failed to create temp file: {}", e)))?;

		// Stream download with progress
		let mut stream = response.bytes_stream();
		let mut last_checkpoint = 0u64;

		while let Some(chunk) = stream.next().await {
			ctx.check_interrupt().await?;

			let chunk = chunk.map_err(|e| JobError::execution(format!("Download error: {}", e)))?;

			file.write_all(&chunk)
				.await
				.map_err(|e| JobError::execution(format!("Write error: {}", e)))?;

			self.state.downloaded_bytes += chunk.len() as u64;

			// Report progress
			ctx.progress(Progress::Bytes {
				current: self.state.downloaded_bytes,
				total: self.state.total_bytes,
			});

			// Checkpoint every 10 MB
			if self.state.downloaded_bytes - last_checkpoint > 10 * 1024 * 1024 {
				ctx.checkpoint().await?;
				last_checkpoint = self.state.downloaded_bytes;
				let progress_pct =
					(self.state.downloaded_bytes as f64 / self.state.total_bytes as f64) * 100.0;
				debug!(
					"Download progress: {:.1}% ({} MB / {} MB)",
					progress_pct,
					self.state.downloaded_bytes / 1024 / 1024,
					self.state.total_bytes / 1024 / 1024
				);
			}
		}

		file.flush().await?;
		drop(file);

		ctx.log(format!(
			"Downloaded {} MB",
			self.state.downloaded_bytes / 1024 / 1024
		));

		Ok(())
	}

	async fn verify(&mut self, ctx: &JobContext<'_>) -> JobResult<()> {
		ctx.log("Verifying download...");

		// Check if this is an archive that needs extraction
		let is_archive = self.state.download_url.ends_with(".tar.gz")
			|| self.state.download_url.ends_with(".zip");

		if is_archive {
			// Extract archive
			ctx.log("Extracting archive...");
			self.extract_archive(ctx).await?;
		} else {
			// Check file size
			let metadata = tokio::fs::metadata(&self.state.temp_path)
				.await
				.map_err(|e| JobError::execution(format!("Failed to read temp file: {}", e)))?;

			if metadata.len() != self.state.total_bytes {
				return Err(JobError::execution(format!(
					"Downloaded file size mismatch: expected {} bytes, got {} bytes",
					self.state.total_bytes,
					metadata.len()
				)));
			}

			// Move to final location
			tokio::fs::rename(&self.state.temp_path, &self.state.target_path)
				.await
				.map_err(|e| JobError::execution(format!("Failed to move file: {}", e)))?;
		}

		ctx.log("Verification complete");

		Ok(())
	}

	async fn extract_archive(&mut self, ctx: &JobContext<'_>) -> JobResult<()> {
		use std::fs::File;
		use std::io::{BufReader, Read};

		let temp_path = self.state.temp_path.clone();
		let target_path = self.state.target_path.clone();

		// Extract in blocking task to avoid blocking async runtime
		tokio::task::spawn_blocking(move || {
			if temp_path.extension().and_then(|s| s.to_str()) == Some("zip") {
				// Extract ZIP (Windows)
				let file = File::open(&temp_path)
					.map_err(|e| JobError::execution(format!("Failed to open zip: {}", e)))?;
				let mut archive = zip::ZipArchive::new(BufReader::new(file))
					.map_err(|e| JobError::execution(format!("Failed to read zip: {}", e)))?;

				// Find the executable in the archive
				for i in 0..archive.len() {
					let mut file = archive.by_index(i).map_err(|e| {
						JobError::execution(format!("Failed to read zip entry: {}", e))
					})?;

					if file.name().ends_with("sharp.exe") || file.name().ends_with("sharp") {
						let mut out_file = File::create(&target_path).map_err(|e| {
							JobError::execution(format!("Failed to create target file: {}", e))
						})?;
						std::io::copy(&mut file, &mut out_file).map_err(|e| {
							JobError::execution(format!("Failed to extract file: {}", e))
						})?;
						break;
					}
				}
			} else {
				// Extract TAR.GZ (macOS/Linux)
				let file = File::open(&temp_path)
					.map_err(|e| JobError::execution(format!("Failed to open archive: {}", e)))?;
				let gz = flate2::read::GzDecoder::new(BufReader::new(file));
				let mut archive = tar::Archive::new(gz);

				// Extract entries
				for entry in archive.entries().map_err(|e| {
					JobError::execution(format!("Failed to read archive entries: {}", e))
				})? {
					let mut entry = entry.map_err(|e| {
						JobError::execution(format!("Failed to read archive entry: {}", e))
					})?;

					let path = entry.path().map_err(|e| {
						JobError::execution(format!("Failed to get entry path: {}", e))
					})?;

					if path.file_name().and_then(|n| n.to_str()) == Some("sharp") {
						let mut out_file = File::create(&target_path).map_err(|e| {
							JobError::execution(format!("Failed to create target file: {}", e))
						})?;
						std::io::copy(&mut entry, &mut out_file).map_err(|e| {
							JobError::execution(format!("Failed to extract file: {}", e))
						})?;

						// Make executable on Unix
						#[cfg(unix)]
						{
							use std::os::unix::fs::PermissionsExt;
							let mut perms = out_file.metadata().unwrap().permissions();
							perms.set_mode(0o755);
							std::fs::set_permissions(&target_path, perms).unwrap();
						}
						break;
					}
				}
			}

			// Clean up temp archive
			std::fs::remove_file(&temp_path)
				.map_err(|e| JobError::execution(format!("Failed to remove temp file: {}", e)))?;

			Ok::<(), JobError>(())
		})
		.await
		.map_err(|e| JobError::execution(format!("Extract task panicked: {}", e)))??;

		ctx.log("Archive extracted successfully");

		Ok(())
	}
}

#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct ModelDownloadOutput {
	pub model_id: String,
	pub path: String,
	pub size_bytes: u64,
}

impl From<ModelDownloadOutput> for JobOutput {
	fn from(output: ModelDownloadOutput) -> Self {
		JobOutput::Custom(serde_json::json!({
			"type": "model_download",
			"model_id": output.model_id,
			"path": output.path,
			"size_mb": output.size_bytes / 1024 / 1024,
		}))
	}
}

impl DynJob for ModelDownloadJob {
	fn job_name(&self) -> &'static str {
		"Model Download"
	}
}

impl From<ModelDownloadJob> for Box<dyn DynJob> {
	fn from(job: ModelDownloadJob) -> Self {
		Box::new(job)
	}
}
