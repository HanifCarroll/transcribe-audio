use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use clap::{ArgAction, Args, CommandFactory, Parser, Subcommand};
use regex::Regex;
use serde::Serialize;
use tempfile::TempDir;

const DEFAULT_MODEL_DIR: &str = "~/.local/share/transcribe-audio/models";
const SUPERWHISPER_SMALL: &str = "~/Library/Application Support/superwhisper/ggml-small.bin";
const MODEL_PRIORITY: &[&str] = &["large-v3-turbo", "medium", "small"];
const SUPPORTED_FORMATS: &[&str] = &["txt", "json", "srt", "vtt"];
const SUPPORTED_AUDIO_EXTENSIONS: &[&str] =
    &["aac", "flac", "m4a", "mp3", "ogg", "opus", "wav", "webm"];

#[derive(Clone, Copy)]
struct ModelMeta {
    name: &'static str,
    file: &'static str,
    url: &'static str,
    note: &'static str,
}

const MODELS: &[ModelMeta] = &[
    ModelMeta {
        name: "large-v3-turbo",
        file: "ggml-large-v3-turbo.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
        note: "Recommended default on this Mac: best accuracy/speed tradeoff.",
    },
    ModelMeta {
        name: "medium",
        file: "ggml-medium.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        note: "Good fallback if large-v3-turbo is unavailable.",
    },
    ModelMeta {
        name: "small",
        file: "ggml-small.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        note: "Fast fallback; lower accuracy on names and technical terms.",
    },
];

#[derive(Parser)]
#[command(
    name = "transcribe-audio",
    about = "Transcribe local audio files with ffmpeg + whisper-cli.",
    after_help = "Shorthand works: transcribe-audio \"/path/to/audio.opus\" --print"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Check tools and installed models.
    Doctor(JsonFlag),
    /// List known and installed models.
    Models(JsonFlag),
    /// Download a known whisper.cpp model.
    DownloadModel(DownloadModelArgs),
    /// Transcribe an audio file.
    Transcribe(TranscribeArgs),
    /// Transcribe multiple audio files and write a run manifest.
    Batch(BatchArgs),
    /// Find likely audio files in a folder.
    Discover(DiscoverArgs),
    /// Combine transcript text files into one Markdown artifact.
    Combine(CombineArgs),
}

#[derive(Args)]
struct JsonFlag {
    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct DownloadModelArgs {
    /// Model name to download.
    #[arg(value_parser = ["large-v3-turbo", "medium", "small"])]
    name: String,
    /// Re-download even if the target exists.
    #[arg(long)]
    force: bool,
    /// Reduce curl output.
    #[arg(long)]
    quiet: bool,
    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct TranscribeArgs {
    /// Audio file to transcribe.
    input: Option<PathBuf>,
    /// Model name, model path, or auto.
    #[arg(long, default_value = "auto")]
    model: String,
    /// Spoken language, such as auto, en, or es.
    #[arg(long, default_value = "auto")]
    language: String,
    /// Comma-separated outputs: txt,json,srt,vtt.
    #[arg(long, default_value = "txt,json")]
    formats: String,
    /// Output directory. Defaults to <input folder>/transcripts.
    #[arg(long)]
    output_dir: Option<PathBuf>,
    /// Output basename without extension.
    #[arg(long)]
    output_name: Option<String>,
    /// Initial Whisper prompt for vocabulary and context.
    #[arg(long)]
    prompt: Option<String>,
    /// Read the initial Whisper prompt from a text file.
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    /// Also write a Markdown transcript next to the normal outputs.
    #[arg(long)]
    markdown: bool,
    /// Keep the converted 16 kHz mono WAV.
    #[arg(long)]
    keep_wav: bool,
    /// Disable GPU in whisper-cli.
    #[arg(long)]
    no_gpu: bool,
    /// Show whisper-cli backend output.
    #[arg(long)]
    verbose: bool,
    /// Print the .txt transcript.
    #[arg(long = "print")]
    print_text: bool,
    /// Emit machine-readable JSON summary.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct BatchArgs {
    /// Audio files to transcribe.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    /// Model name, model path, or auto.
    #[arg(long, default_value = "auto")]
    model: String,
    /// Spoken language, such as auto, en, or es.
    #[arg(long, default_value = "auto")]
    language: String,
    /// Comma-separated Whisper outputs: txt,json,srt,vtt.
    #[arg(long, default_value = "txt,json")]
    formats: String,
    /// Run output directory. Defaults to ./transcripts/run-<unix-seconds>.
    #[arg(long)]
    output_dir: Option<PathBuf>,
    /// Initial Whisper prompt for vocabulary and context.
    #[arg(long)]
    prompt: Option<String>,
    /// Read the initial Whisper prompt from a text file.
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    /// Keep converted 16 kHz mono WAV files.
    #[arg(long)]
    keep_wav: bool,
    /// Disable GPU in whisper-cli.
    #[arg(long)]
    no_gpu: bool,
    /// Show whisper-cli backend output.
    #[arg(long)]
    verbose: bool,
    /// Write transcripts.md for successful .txt outputs.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    markdown: bool,
    /// Stop at the first failed input.
    #[arg(long)]
    fail_fast: bool,
    /// Emit machine-readable JSON summary.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct DiscoverArgs {
    /// Folder to scan. Defaults to ~/Downloads.
    root: Option<PathBuf>,
    /// Only include files modified within this age, such as 2d, 6h, or 30m.
    #[arg(long)]
    since: Option<String>,
    /// Maximum number of files to return.
    #[arg(long)]
    limit: Option<usize>,
    /// Recurse into subdirectories.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    recursive: bool,
    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct CombineArgs {
    /// Transcript .txt files or directories containing .txt files.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    /// Markdown output path. Defaults to transcripts.md.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Markdown title.
    #[arg(long, default_value = "Transcripts")]
    title: String,
    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Serialize)]
struct ModelRow {
    name: String,
    file: String,
    installed: bool,
    paths: Vec<String>,
    note: String,
}

#[derive(Serialize)]
struct DoctorPayload {
    ok: bool,
    tools: ToolsPayload,
    paths: PathsPayload,
    models: Vec<ModelRow>,
}

#[derive(Serialize)]
struct ToolsPayload {
    ffmpeg: Option<String>,
    #[serde(rename = "whisper_cli")]
    whisper_cli: Option<String>,
    curl: Option<String>,
}

#[derive(Serialize)]
struct PathsPayload {
    #[serde(rename = "model_dir")]
    model_dir: String,
    #[serde(rename = "superwhisper_small")]
    superwhisper_small: String,
}

#[derive(Serialize)]
struct ModelsPayload {
    ok: bool,
    recommended: String,
    models: Vec<ModelRow>,
}

#[derive(Serialize)]
struct DownloadPayload {
    ok: bool,
    downloaded: bool,
    model: String,
    path: String,
}

#[derive(Serialize)]
struct TranscribePayload {
    ok: bool,
    input: String,
    model: ModelSelectionPayload,
    language: String,
    formats: Vec<String>,
    outputs: BTreeMap<String, String>,
    duration_seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    converted_wav: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    markdown: Option<String>,
}

#[derive(Serialize)]
struct ModelSelectionPayload {
    name: String,
    path: String,
}

#[derive(Serialize)]
struct ErrorPayload {
    ok: bool,
    stage: String,
    error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    command: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exit_code: Option<i32>,
}

#[derive(Serialize)]
struct BatchPayload {
    ok: bool,
    run_dir: String,
    manifest_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    markdown: Option<String>,
    input_count: usize,
    success_count: usize,
    failure_count: usize,
    items: Vec<BatchItem>,
}

#[derive(Serialize)]
struct BatchItem {
    ok: bool,
    input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    outputs: Option<BTreeMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<ErrorPayload>,
}

#[derive(Serialize)]
struct DiscoverPayload {
    ok: bool,
    root: String,
    count: usize,
    files: Vec<DiscoveredFile>,
}

#[derive(Serialize)]
struct DiscoveredFile {
    path: String,
    modified_unix_seconds: Option<u64>,
    size_bytes: u64,
}

#[derive(Serialize)]
struct CombinePayload {
    ok: bool,
    output: String,
    input_count: usize,
    inputs: Vec<String>,
}

#[derive(Debug)]
struct StageError {
    stage: &'static str,
    message: String,
    command: Option<String>,
    exit_code: Option<i32>,
}

struct EnvPaths {
    model_dir: PathBuf,
    superwhisper_small: PathBuf,
    ffmpeg: Option<PathBuf>,
    whisper_cli: Option<PathBuf>,
    curl: Option<PathBuf>,
}

impl StageError {
    fn new(stage: &'static str, message: impl Into<String>) -> Self {
        Self {
            stage,
            message: message.into(),
            command: None,
            exit_code: None,
        }
    }

    fn command(
        stage: &'static str,
        message: impl Into<String>,
        command: impl Into<String>,
        exit_code: Option<i32>,
    ) -> Self {
        Self {
            stage,
            message: message.into(),
            command: Some(command.into()),
            exit_code,
        }
    }
}

impl std::fmt::Display for StageError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for StageError {}

fn error_payload(err: &anyhow::Error) -> ErrorPayload {
    if let Some(stage_error) = err.downcast_ref::<StageError>() {
        ErrorPayload {
            ok: false,
            stage: stage_error.stage.to_string(),
            error: stage_error.message.clone(),
            command: stage_error.command.clone(),
            exit_code: stage_error.exit_code,
        }
    } else {
        ErrorPayload {
            ok: false,
            stage: "cli".to_string(),
            error: err.to_string(),
            command: None,
            exit_code: None,
        }
    }
}

fn main() {
    let raw_args: Vec<OsString> = env::args_os().skip(1).collect();
    if raw_args.is_empty() {
        Cli::command().print_help().unwrap();
        println!();
        std::process::exit(1);
    }
    let json_on_error = raw_args.iter().any(|arg| arg == "--json");
    let cli_args = normalize_shorthand(raw_args);

    let result = Cli::try_parse_from(cli_args)
        .map_err(anyhow::Error::from)
        .and_then(run_cli);

    if let Err(err) = result {
        if json_on_error {
            let payload = error_payload(&err);
            println!("{}", serde_json::to_string_pretty(&payload).unwrap());
        } else {
            eprintln!("transcribe-audio: {err}");
        }
        std::process::exit(1);
    }
}

fn normalize_shorthand(raw_args: Vec<OsString>) -> Vec<OsString> {
    let mut args = Vec::with_capacity(raw_args.len() + 2);
    args.push(OsString::from("transcribe-audio"));
    if let Some(first) = raw_args.first() {
        let commands = [
            "doctor",
            "models",
            "download-model",
            "transcribe",
            "batch",
            "discover",
            "combine",
        ];
        let first = first.to_string_lossy();
        if first != "-h" && first != "--help" && !commands.iter().any(|cmd| first.as_ref() == *cmd)
        {
            args.push(OsString::from("transcribe"));
        }
    }
    args.extend(raw_args);
    args
}

fn run_cli(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Doctor(args) => run_doctor(args),
        Commands::Models(args) => run_models(args),
        Commands::DownloadModel(args) => run_download_model(args),
        Commands::Transcribe(args) => run_transcribe_command(args),
        Commands::Batch(args) => run_batch_command(args),
        Commands::Discover(args) => run_discover_command(args),
        Commands::Combine(args) => run_combine_command(args),
    }
}

fn env_paths() -> EnvPaths {
    let model_dir = env::var_os("TRANSCRIBE_AUDIO_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));
    let ffmpeg = env_tool("FFMPEG", "ffmpeg");
    let whisper_cli = env_tool("WHISPER_CLI", "whisper-cli");
    let curl = env_tool("CURL", "curl");
    EnvPaths {
        model_dir: expand_tilde(&model_dir),
        superwhisper_small: expand_tilde(Path::new(SUPERWHISPER_SMALL)),
        ffmpeg,
        whisper_cli,
        curl,
    }
}

fn env_tool(env_name: &str, command: &str) -> Option<PathBuf> {
    env::var_os(env_name)
        .map(PathBuf::from)
        .or_else(|| which::which(command).ok())
        .map(|path| expand_tilde(&path))
}

fn expand_tilde(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if raw == "~" {
        home_dir().unwrap_or_else(|| path.to_path_buf())
    } else if let Some(rest) = raw.strip_prefix("~/") {
        home_dir()
            .map(|home| home.join(rest))
            .unwrap_or_else(|| path.to_path_buf())
    } else {
        path.to_path_buf()
    }
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME").map(PathBuf::from)
}

fn run_doctor(args: JsonFlag) -> Result<()> {
    let env_paths = env_paths();
    let models = installed_models(&env_paths);
    let payload = DoctorPayload {
        ok: env_paths.ffmpeg.is_some()
            && env_paths.whisper_cli.is_some()
            && models.iter().any(|model| model.installed),
        tools: ToolsPayload {
            ffmpeg: env_paths.ffmpeg.as_ref().map(|path| path_string(path)),
            whisper_cli: env_paths.whisper_cli.as_ref().map(|path| path_string(path)),
            curl: env_paths.curl.as_ref().map(|path| path_string(path)),
        },
        paths: PathsPayload {
            model_dir: path_string(&env_paths.model_dir),
            superwhisper_small: path_string(&env_paths.superwhisper_small),
        },
        models,
    };

    if args.json {
        print_json(&payload)?;
    } else {
        println!(
            "ffmpeg: {}",
            payload.tools.ffmpeg.as_deref().unwrap_or("missing")
        );
        println!(
            "whisper-cli: {}",
            payload.tools.whisper_cli.as_deref().unwrap_or("missing")
        );
        println!("model dir: {}", payload.paths.model_dir);
        for row in &payload.models {
            let status = if row.installed {
                "installed"
            } else {
                "missing"
            };
            println!("{}: {}", row.name, status);
            for path in &row.paths {
                println!("  {path}");
            }
        }
    }
    Ok(())
}

fn run_models(args: JsonFlag) -> Result<()> {
    let payload = ModelsPayload {
        ok: true,
        recommended: "large-v3-turbo".to_string(),
        models: installed_models(&env_paths()),
    };

    if args.json {
        print_json(&payload)?;
    } else {
        println!("Recommended: large-v3-turbo");
        for row in &payload.models {
            let status = if row.installed {
                "installed"
            } else {
                "missing"
            };
            println!("- {} ({}): {}", row.name, status, row.note);
            for path in &row.paths {
                println!("  {path}");
            }
        }
    }
    Ok(())
}

fn run_download_model(args: DownloadModelArgs) -> Result<()> {
    let env_paths = env_paths();
    let meta = model_meta(&args.name).ok_or_else(|| anyhow!("unknown model: {}", args.name))?;
    let curl = env_paths
        .curl
        .as_ref()
        .ok_or_else(|| StageError::new("download", "curl is required to download models"))?;

    fs::create_dir_all(&env_paths.model_dir).with_context(|| {
        format!(
            "failed to create model directory: {}",
            env_paths.model_dir.display()
        )
    })?;
    let target = env_paths.model_dir.join(meta.file);
    let part = PathBuf::from(format!("{}.part", target.display()));

    if target.exists() && !args.force {
        let payload = DownloadPayload {
            ok: true,
            downloaded: false,
            model: args.name,
            path: path_string(&target),
        };
        if args.json {
            print_json(&payload)?;
        } else {
            println!("Already installed: {}", target.display());
        }
        return Ok(());
    }

    let mut cmd = Command::new(curl);
    if args.quiet {
        cmd.arg("--silent").arg("--show-error");
    }
    cmd.arg("--fail")
        .arg("--location")
        .arg("--continue-at")
        .arg("-")
        .arg("--output")
        .arg(&part)
        .arg(meta.url);
    run_command(&mut cmd, "download", false)?;
    fs::rename(&part, &target).with_context(|| {
        format!(
            "failed to move downloaded model from {} to {}",
            part.display(),
            target.display()
        )
    })?;

    let payload = DownloadPayload {
        ok: true,
        downloaded: true,
        model: args.name,
        path: path_string(&target),
    };
    if args.json {
        print_json(&payload)?;
    } else {
        println!("Installed {}: {}", payload.model, payload.path);
    }
    Ok(())
}

fn run_transcribe_command(args: TranscribeArgs) -> Result<()> {
    let input = args.input.as_ref().ok_or_else(|| {
        StageError::new(
            "cli",
            "missing audio input. Run `transcribe-audio --help` for usage.",
        )
    })?;
    let payload = transcribe(&args, input)?;

    if args.json {
        print_json(&payload)?;
    } else {
        println!("Model: {}", payload.model.name);
        for (fmt, path) in &payload.outputs {
            println!("{fmt}: {path}");
        }
        println!("Done in {}s", payload.duration_seconds);
    }

    if args.print_text
        && let Some(txt) = payload.outputs.get("txt")
    {
        let content = fs::read_to_string(txt)
            .with_context(|| format!("failed to read transcript text: {txt}"))?;
        println!("\n{content}");
    }
    Ok(())
}

fn transcribe(args: &TranscribeArgs, input: &Path) -> Result<TranscribePayload> {
    let env_paths = env_paths();
    ensure_tools(&env_paths)?;

    let input_path = expand_tilde(input).canonicalize().map_err(|_| {
        StageError::new(
            "input",
            format!("input file does not exist: {}", input.display()),
        )
    })?;
    let model = choose_model(&args.model, &env_paths)?;
    let formats = parse_formats(&args.formats)?;
    let prompt = resolve_prompt(args.prompt.as_deref(), args.prompt_file.as_deref())?;

    let output_dir = match &args.output_dir {
        Some(dir) => expand_tilde(dir),
        None => input_path
            .parent()
            .map(|parent| parent.join("transcripts"))
            .unwrap_or_else(|| PathBuf::from("transcripts")),
    };
    fs::create_dir_all(&output_dir).with_context(|| {
        format!(
            "failed to create output directory: {}",
            output_dir.display()
        )
    })?;

    let output_name = args.output_name.clone().unwrap_or_else(|| {
        slugify(
            input_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .as_ref(),
        )
    });
    let output_base = output_dir.join(output_name);

    let start = Instant::now();
    let tmp_dir = TempDir::new().context("failed to create temporary directory")?;
    let tmp_wav = tmp_dir.path().join(format!(
        "{}.wav",
        output_base
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
    ));
    convert_audio(&env_paths, &input_path, &tmp_wav)?;

    let converted_wav = if args.keep_wav {
        let kept = output_path_for_format(&output_base, "wav");
        fs::copy(&tmp_wav, &kept).with_context(|| {
            format!(
                "failed to copy converted WAV from {} to {}",
                tmp_wav.display(),
                kept.display()
            )
        })?;
        Some(path_string(&kept))
    } else {
        None
    };

    let whisper_cli = env_paths
        .whisper_cli
        .as_ref()
        .expect("checked by ensure_tools");
    let mut cmd = Command::new(whisper_cli);
    cmd.arg("-m")
        .arg(&model.path)
        .arg("-f")
        .arg(&tmp_wav)
        .arg("-l")
        .arg(&args.language)
        .arg("-of")
        .arg(&output_base)
        .arg("--no-prints");
    if let Some(prompt) = &prompt {
        cmd.arg("--prompt").arg(prompt);
    }
    if args.no_gpu {
        cmd.arg("--no-gpu");
    }
    if formats.iter().any(|fmt| fmt == "txt") {
        cmd.arg("-otxt");
    }
    if formats.iter().any(|fmt| fmt == "json") {
        cmd.arg("-oj");
    }
    if formats.iter().any(|fmt| fmt == "srt") {
        cmd.arg("-osrt");
    }
    if formats.iter().any(|fmt| fmt == "vtt") {
        cmd.arg("-ovtt");
    }
    run_command(&mut cmd, "whisper", !args.verbose)?;

    let outputs = wait_for_outputs(&output_base, &formats, Duration::from_secs(10));
    let markdown = if args.markdown {
        let txt_path = outputs
            .get("txt")
            .ok_or_else(|| StageError::new("outputs", "--markdown requires txt output format"))?;
        let markdown_path = output_path_for_format(&output_base, "md");
        write_combined_markdown(
            &[(input_label(&input_path), PathBuf::from(txt_path))],
            &markdown_path,
            "Transcript",
        )?;
        Some(path_string(&markdown_path))
    } else {
        None
    };
    let elapsed = (start.elapsed().as_secs_f64() * 1000.0).round() / 1000.0;

    Ok(TranscribePayload {
        ok: true,
        input: path_string(&input_path),
        model: ModelSelectionPayload {
            name: model.name,
            path: path_string(&model.path),
        },
        language: args.language.clone(),
        formats,
        outputs,
        duration_seconds: elapsed,
        converted_wav,
        markdown,
    })
}

fn run_batch_command(args: BatchArgs) -> Result<()> {
    let emit_json = args.json;
    let payload = batch(args)?;
    if !emit_json {
        if payload.ok {
            println!(
                "Transcribed {} files. Manifest: {}",
                payload.success_count, payload.manifest_path
            );
        } else {
            println!(
                "Transcribed {}/{} files with {} failure(s). Manifest: {}",
                payload.success_count,
                payload.input_count,
                payload.failure_count,
                payload.manifest_path
            );
        }
        if let Some(markdown) = &payload.markdown {
            println!("Markdown: {markdown}");
        }
        if payload.items.iter().any(|item| !item.ok) {
            for item in &payload.items {
                if let Some(error) = &item.error {
                    eprintln!("failed: {}: {}", item.input, error.error);
                }
            }
        }
    }
    Ok(())
}

fn batch(args: BatchArgs) -> Result<BatchPayload> {
    let run_dir = match &args.output_dir {
        Some(dir) => expand_tilde(dir),
        None => env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("transcripts")
            .join(format!("run-{}", unix_now_seconds())),
    };
    fs::create_dir_all(&run_dir).map_err(|err| {
        StageError::new(
            "outputs",
            format!(
                "failed to create run directory {}: {err}",
                run_dir.display()
            ),
        )
    })?;

    let formats = parse_formats(&args.formats)?;
    let mut items = Vec::new();
    let mut successful_txt = Vec::new();

    for (index, input) in args.inputs.iter().enumerate() {
        let output_name = unique_output_name(input, index + 1, &run_dir);
        let transcribe_args = TranscribeArgs {
            input: Some(input.clone()),
            model: args.model.clone(),
            language: args.language.clone(),
            formats: formats.join(","),
            output_dir: Some(run_dir.clone()),
            output_name: Some(output_name.clone()),
            prompt: args.prompt.clone(),
            prompt_file: args.prompt_file.clone(),
            markdown: false,
            keep_wav: args.keep_wav,
            no_gpu: args.no_gpu,
            verbose: args.verbose,
            print_text: false,
            json: true,
        };

        match transcribe(&transcribe_args, input) {
            Ok(payload) => {
                if let Some(txt) = payload.outputs.get("txt") {
                    successful_txt
                        .push((input_label(Path::new(&payload.input)), PathBuf::from(txt)));
                }
                items.push(BatchItem {
                    ok: true,
                    input: payload.input,
                    output_name: Some(output_name),
                    outputs: Some(payload.outputs),
                    duration_seconds: Some(payload.duration_seconds),
                    error: None,
                });
            }
            Err(err) => {
                let error = error_payload(&err);
                items.push(BatchItem {
                    ok: false,
                    input: path_string(&expand_tilde(input)),
                    output_name: Some(output_name),
                    outputs: None,
                    duration_seconds: None,
                    error: Some(error),
                });
                if args.fail_fast {
                    break;
                }
            }
        }
    }

    let markdown = if args.markdown && !successful_txt.is_empty() {
        let markdown_path = run_dir.join("transcripts.md");
        write_combined_markdown(&successful_txt, &markdown_path, "Transcripts")?;
        Some(path_string(&markdown_path))
    } else {
        None
    };
    let success_count = items.iter().filter(|item| item.ok).count();
    let failure_count = items.len().saturating_sub(success_count);
    let manifest_path = run_dir.join("run.json");
    let payload = BatchPayload {
        ok: failure_count == 0,
        run_dir: path_string(&run_dir),
        manifest_path: path_string(&manifest_path),
        markdown,
        input_count: items.len(),
        success_count,
        failure_count,
        items,
    };
    write_json_file(&manifest_path, &payload)?;

    if args.json {
        print_json(&payload)?;
    }
    Ok(payload)
}

fn run_discover_command(args: DiscoverArgs) -> Result<()> {
    let emit_json = args.json;
    let payload = discover(args)?;
    if !emit_json {
        if payload.files.is_empty() {
            println!("No audio files found.");
        } else {
            for file in &payload.files {
                println!("{}", file.path);
            }
        }
    }
    Ok(())
}

fn discover(args: DiscoverArgs) -> Result<DiscoverPayload> {
    let root = args
        .root
        .map(|path| expand_tilde(&path))
        .unwrap_or_else(|| expand_tilde(Path::new("~/Downloads")));
    let root = root.canonicalize().map_err(|err| {
        StageError::new(
            "discover",
            format!("failed to read discovery root {}: {err}", root.display()),
        )
    })?;
    let since = match &args.since {
        Some(raw) => Some(parse_age(raw)?),
        None => None,
    };
    let min_modified = since.and_then(|age| SystemTime::now().checked_sub(age));
    let mut files = Vec::new();
    collect_audio_files(&root, args.recursive, min_modified, &mut files)?;
    files.sort_by(|a, b| b.modified_unix_seconds.cmp(&a.modified_unix_seconds));
    if let Some(limit) = args.limit {
        files.truncate(limit);
    }
    let payload = DiscoverPayload {
        ok: true,
        root: path_string(&root),
        count: files.len(),
        files,
    };
    if args.json {
        print_json(&payload)?;
    }
    Ok(payload)
}

fn run_combine_command(args: CombineArgs) -> Result<()> {
    let emit_json = args.json;
    let payload = combine(args)?;
    if !emit_json {
        println!("Wrote {}", payload.output);
    }
    Ok(())
}

fn combine(args: CombineArgs) -> Result<CombinePayload> {
    let inputs = collect_transcript_inputs(&args.inputs)?;
    let output = args
        .output
        .map(|path| expand_tilde(&path))
        .unwrap_or_else(|| PathBuf::from("transcripts.md"));
    write_combined_markdown(&inputs, &output, &args.title)?;
    let payload = CombinePayload {
        ok: true,
        output: path_string(&output),
        input_count: inputs.len(),
        inputs: inputs.iter().map(|(_, path)| path_string(path)).collect(),
    };
    if args.json {
        print_json(&payload)?;
    }
    Ok(payload)
}

fn installed_models(env_paths: &EnvPaths) -> Vec<ModelRow> {
    MODELS
        .iter()
        .map(|meta| {
            let mut candidates = vec![env_paths.model_dir.join(meta.file)];
            if meta.name == "small" {
                candidates.push(env_paths.superwhisper_small.clone());
            }
            let paths: Vec<String> = candidates
                .into_iter()
                .filter(|path| path.exists())
                .map(|path| path_string(&path))
                .collect();
            ModelRow {
                name: meta.name.to_string(),
                file: meta.file.to_string(),
                installed: !paths.is_empty(),
                paths,
                note: meta.note.to_string(),
            }
        })
        .collect()
}

struct SelectedModel {
    name: String,
    path: PathBuf,
}

fn choose_model(model_arg: &str, env_paths: &EnvPaths) -> Result<SelectedModel> {
    if model_arg == "auto" {
        if let Ok(env_model) = env::var("TRANSCRIBE_AUDIO_MODEL") {
            return model_path_for(&env_model, env_paths).ok_or_else(|| {
                anyhow!("TRANSCRIBE_AUDIO_MODEL does not exist or is not installed: {env_model}")
            });
        }
        for name in MODEL_PRIORITY {
            if let Some(model) = model_path_for(name, env_paths) {
                return Ok(model);
            }
        }
        return Err(StageError::new(
            "model",
            "no Whisper model found. Run `transcribe-audio download-model large-v3-turbo` or set TRANSCRIBE_AUDIO_MODEL=/path/to/ggml-model.bin."
        )
        .into());
    }

    if let Some(model) = model_path_for(model_arg, env_paths) {
        return Ok(model);
    }
    let known = MODELS
        .iter()
        .map(|model| model.name)
        .collect::<Vec<_>>()
        .join(", ");
    Err(StageError::new(
        "model",
        "model is not installed: {model_arg}. Run `transcribe-audio download-model {model_arg}` for known models ({known}), or pass a model file path."
            .replace("{model_arg}", model_arg)
            .replace("{known}", &known),
    )
    .into())
}

fn model_path_for(name_or_path: &str, env_paths: &EnvPaths) -> Option<SelectedModel> {
    let expanded = expand_tilde(Path::new(name_or_path));
    if expanded.exists() {
        return Some(SelectedModel {
            name: expanded
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            path: expanded,
        });
    }

    let meta = model_meta(name_or_path)?;
    let candidate = env_paths.model_dir.join(meta.file);
    if candidate.exists() {
        return Some(SelectedModel {
            name: meta.name.to_string(),
            path: candidate,
        });
    }
    if meta.name == "small" && env_paths.superwhisper_small.exists() {
        return Some(SelectedModel {
            name: "small".to_string(),
            path: env_paths.superwhisper_small.clone(),
        });
    }
    None
}

fn model_meta(name: &str) -> Option<ModelMeta> {
    MODELS.iter().copied().find(|meta| meta.name == name)
}

fn ensure_tools(env_paths: &EnvPaths) -> Result<()> {
    let mut missing = Vec::new();
    if env_paths.ffmpeg.is_none() {
        missing.push("ffmpeg");
    }
    if env_paths.whisper_cli.is_none() {
        missing.push("whisper-cli");
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(StageError::new(
            "doctor",
            format!("missing required tool(s): {}", missing.join(", ")),
        )
        .into())
    }
}

fn convert_audio(env_paths: &EnvPaths, input_path: &Path, output_wav: &Path) -> Result<()> {
    let ffmpeg = env_paths
        .ffmpeg
        .as_ref()
        .ok_or_else(|| StageError::new("convert", "ffmpeg is required for audio conversion"))?;
    let mut cmd = Command::new(ffmpeg);
    cmd.arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-y")
        .arg("-i")
        .arg(input_path)
        .arg("-ar")
        .arg("16000")
        .arg("-ac")
        .arg("1")
        .arg(output_wav);
    run_command(&mut cmd, "convert", false)
}

fn run_command(cmd: &mut Command, stage: &'static str, quiet: bool) -> Result<()> {
    if quiet {
        cmd.stdout(Stdio::null()).stderr(Stdio::null());
    }
    let printable = format!("{cmd:?}");
    let status = cmd.status().map_err(|err| {
        StageError::command(
            stage,
            format!("failed to start command: {err}"),
            &printable,
            None,
        )
    })?;
    if status.success() {
        Ok(())
    } else {
        Err(StageError::command(
            stage,
            format!(
                "command failed ({}): {printable}",
                status.code().unwrap_or(-1)
            ),
            printable,
            status.code(),
        )
        .into())
    }
}

fn parse_formats(raw: &str) -> Result<Vec<String>> {
    let mut values = Vec::new();
    for value in raw.split(',').map(|part| part.trim().to_lowercase()) {
        if value.is_empty() {
            continue;
        }
        if !SUPPORTED_FORMATS.contains(&value.as_str()) {
            return Err(
                StageError::new("cli", format!("unsupported output format(s): {value}")).into(),
            );
        }
        if !values.contains(&value) {
            values.push(value);
        }
    }
    if values.is_empty() {
        Ok(vec!["txt".to_string(), "json".to_string()])
    } else {
        Ok(values)
    }
}

fn resolve_prompt(prompt: Option<&str>, prompt_file: Option<&Path>) -> Result<Option<String>> {
    let file_prompt = match prompt_file {
        Some(path) => {
            let path = expand_tilde(path);
            Some(
                fs::read_to_string(&path)
                    .map_err(|err| {
                        StageError::new(
                            "prompt",
                            format!("failed to read prompt file {}: {err}", path.display()),
                        )
                    })?
                    .trim()
                    .to_string(),
            )
        }
        None => None,
    };

    Ok(match (file_prompt, prompt) {
        (Some(file_prompt), Some(prompt)) if !prompt.trim().is_empty() => {
            Some(format!("{}\n\n{}", file_prompt, prompt.trim()))
        }
        (Some(file_prompt), _) if !file_prompt.is_empty() => Some(file_prompt),
        (_, Some(prompt)) if !prompt.trim().is_empty() => Some(prompt.trim().to_string()),
        _ => None,
    })
}

fn slugify(value: &str) -> String {
    let stem = Path::new(value)
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let invalid = Regex::new(r"[^A-Za-z0-9._-]+").unwrap();
    let repeated_dash = Regex::new(r"-{2,}").unwrap();
    let slug = invalid.replace_all(stem.trim(), "-");
    let slug = repeated_dash.replace_all(&slug, "-");
    let slug = slug.trim_matches(['-', '.', '_']).to_string();
    if slug.is_empty() {
        "transcript".to_string()
    } else {
        slug
    }
}

fn collect_outputs(output_base: &Path, formats: &[String]) -> BTreeMap<String, String> {
    formats
        .iter()
        .filter_map(|fmt| {
            let path = output_path_for_format(output_base, fmt);
            path.exists().then(|| (fmt.clone(), path_string(&path)))
        })
        .collect()
}

fn wait_for_outputs(
    output_base: &Path,
    formats: &[String],
    timeout: Duration,
) -> BTreeMap<String, String> {
    let deadline = Instant::now() + timeout;
    let mut outputs = collect_outputs(output_base, formats);
    while outputs.len() < formats.len() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(50));
        outputs = collect_outputs(output_base, formats);
    }
    outputs
}

fn output_path_for_format(output_base: &Path, fmt: &str) -> PathBuf {
    PathBuf::from(format!("{}.{}", output_base.display(), fmt))
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

fn print_json<T: Serialize>(payload: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(payload)?);
    Ok(())
}

fn write_json_file<T: Serialize>(path: &Path, payload: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            StageError::new(
                "outputs",
                format!(
                    "failed to create output directory {}: {err}",
                    parent.display()
                ),
            )
        })?;
    }
    let json = serde_json::to_string_pretty(payload)?;
    fs::write(path, format!("{json}\n")).map_err(|err| {
        StageError::new(
            "outputs",
            format!("failed to write JSON file {}: {err}", path.display()),
        )
    })?;
    Ok(())
}

fn write_combined_markdown(inputs: &[(String, PathBuf)], output: &Path, title: &str) -> Result<()> {
    if inputs.is_empty() {
        return Err(StageError::new("combine", "no transcript inputs to combine").into());
    }
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|err| {
            StageError::new(
                "outputs",
                format!(
                    "failed to create output directory {}: {err}",
                    parent.display()
                ),
            )
        })?;
    }

    let mut markdown = String::new();
    markdown.push_str("# ");
    markdown.push_str(title.trim());
    markdown.push_str("\n\n");
    for (label, path) in inputs {
        let content = fs::read_to_string(path).map_err(|err| {
            StageError::new(
                "combine",
                format!("failed to read transcript {}: {err}", path.display()),
            )
        })?;
        markdown.push_str("## ");
        markdown.push_str(label.trim());
        markdown.push_str("\n\n");
        markdown.push_str(content.trim());
        markdown.push_str("\n\n");
    }
    fs::write(output, markdown).map_err(|err| {
        StageError::new(
            "outputs",
            format!("failed to write markdown file {}: {err}", output.display()),
        )
    })?;
    Ok(())
}

fn collect_transcript_inputs(inputs: &[PathBuf]) -> Result<Vec<(String, PathBuf)>> {
    let mut transcripts = Vec::new();
    for input in inputs {
        let path = expand_tilde(input);
        if path.is_dir() {
            let mut dir_entries = Vec::new();
            for entry in fs::read_dir(&path).map_err(|err| {
                StageError::new(
                    "combine",
                    format!("failed to read directory {}: {err}", path.display()),
                )
            })? {
                let entry = entry.map_err(|err| {
                    StageError::new(
                        "combine",
                        format!(
                            "failed to read directory entry in {}: {err}",
                            path.display()
                        ),
                    )
                })?;
                let entry_path = entry.path();
                if entry_path.extension().and_then(|ext| ext.to_str()) == Some("txt") {
                    dir_entries.push(entry_path);
                }
            }
            dir_entries.sort();
            for entry_path in dir_entries {
                transcripts.push((input_label(&entry_path), entry_path));
            }
        } else if path.exists() {
            transcripts.push((input_label(&path), path));
        } else {
            return Err(StageError::new(
                "combine",
                format!("transcript input does not exist: {}", path.display()),
            )
            .into());
        }
    }
    if transcripts.is_empty() {
        return Err(StageError::new("combine", "no transcript .txt files found").into());
    }
    Ok(transcripts)
}

fn collect_audio_files(
    root: &Path,
    recursive: bool,
    min_modified: Option<SystemTime>,
    files: &mut Vec<DiscoveredFile>,
) -> Result<()> {
    for entry in fs::read_dir(root).map_err(|err| {
        StageError::new(
            "discover",
            format!("failed to read directory {}: {err}", root.display()),
        )
    })? {
        let entry = entry.map_err(|err| {
            StageError::new(
                "discover",
                format!(
                    "failed to read directory entry in {}: {err}",
                    root.display()
                ),
            )
        })?;
        let path = entry.path();
        let metadata = entry.metadata().map_err(|err| {
            StageError::new(
                "discover",
                format!("failed to read metadata for {}: {err}", path.display()),
            )
        })?;
        if metadata.is_dir() {
            if recursive && !is_hidden_path(&path) {
                collect_audio_files(&path, recursive, min_modified, files)?;
            }
            continue;
        }
        if !metadata.is_file() || !is_supported_audio_path(&path) {
            continue;
        }
        let modified = metadata.modified().ok();
        if let (Some(min_modified), Some(modified)) = (min_modified, modified)
            && modified < min_modified
        {
            continue;
        }
        files.push(DiscoveredFile {
            path: path_string(&path),
            modified_unix_seconds: modified.and_then(system_time_to_unix_seconds),
            size_bytes: metadata.len(),
        });
    }
    Ok(())
}

fn is_supported_audio_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            SUPPORTED_AUDIO_EXTENSIONS
                .iter()
                .any(|supported| ext.eq_ignore_ascii_case(supported))
        })
        .unwrap_or(false)
}

fn is_hidden_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.starts_with('.'))
        .unwrap_or(false)
}

fn parse_age(raw: &str) -> Result<Duration> {
    let value = raw.trim();
    let captures = Regex::new(r"^(\d+)([smhdw])$").unwrap();
    let captures = captures.captures(value).ok_or_else(|| {
        StageError::new(
            "cli",
            format!("invalid --since value: {value}. Use formats like 30m, 6h, 2d, or 1w."),
        )
    })?;
    let amount: u64 = captures[1].parse().map_err(|err| {
        StageError::new("cli", format!("invalid --since amount in {value}: {err}"))
    })?;
    let seconds = match &captures[2] {
        "s" => amount,
        "m" => amount * 60,
        "h" => amount * 60 * 60,
        "d" => amount * 60 * 60 * 24,
        "w" => amount * 60 * 60 * 24 * 7,
        _ => unreachable!(),
    };
    Ok(Duration::from_secs(seconds))
}

fn unique_output_name(input: &Path, index: usize, output_dir: &Path) -> String {
    let base = slugify(
        input
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .as_ref(),
    );
    if !output_path_for_format(&output_dir.join(&base), "txt").exists() {
        return base;
    }
    format!("{index:03}-{base}")
}

fn input_label(path: &Path) -> String {
    path.file_stem()
        .or_else(|| path.file_name())
        .unwrap_or_default()
        .to_string_lossy()
        .to_string()
}

fn unix_now_seconds() -> u64 {
    system_time_to_unix_seconds(SystemTime::now()).unwrap_or(0)
}

fn system_time_to_unix_seconds(time: SystemTime) -> Option<u64> {
    time.duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_keeps_dotted_whatsapp_times() {
        assert_eq!(
            slugify("WhatsApp Audio 2026-04-20 at 07.50.30.opus"),
            "WhatsApp-Audio-2026-04-20-at-07.50.30"
        );
    }

    #[test]
    fn output_path_appends_format_instead_of_replacing_dotted_suffix() {
        let base = PathBuf::from("/tmp/WhatsApp-Audio-07.50.30");
        assert_eq!(
            output_path_for_format(&base, "txt"),
            PathBuf::from("/tmp/WhatsApp-Audio-07.50.30.txt")
        );
    }

    #[test]
    fn parse_formats_defaults_and_deduplicates() {
        assert_eq!(
            parse_formats(" txt, json,txt ").unwrap(),
            vec!["txt".to_string(), "json".to_string()]
        );
        assert_eq!(
            parse_formats("").unwrap(),
            vec!["txt".to_string(), "json".to_string()]
        );
    }

    #[test]
    fn parse_age_accepts_agent_friendly_units() {
        assert_eq!(parse_age("30m").unwrap(), Duration::from_secs(30 * 60));
        assert_eq!(
            parse_age("2d").unwrap(),
            Duration::from_secs(2 * 24 * 60 * 60)
        );
    }

    #[test]
    fn supported_audio_extensions_are_case_insensitive() {
        assert!(is_supported_audio_path(Path::new("voice.OPUS")));
        assert!(is_supported_audio_path(Path::new("clip.webm")));
        assert!(!is_supported_audio_path(Path::new("notes.txt")));
    }

    #[test]
    fn resolve_prompt_combines_prompt_file_and_inline_prompt() {
        let temp_dir = TempDir::new().unwrap();
        let prompt_path = temp_dir.path().join("prompt.txt");
        fs::write(&prompt_path, "Client vocabulary").unwrap();
        let prompt = resolve_prompt(Some("Names matter"), Some(&prompt_path)).unwrap();
        assert_eq!(
            prompt,
            Some("Client vocabulary\n\nNames matter".to_string())
        );
    }
}
