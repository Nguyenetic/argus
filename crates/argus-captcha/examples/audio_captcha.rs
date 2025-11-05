/// Audio CAPTCHA solver example
///
/// This example demonstrates solving audio CAPTCHAs using Whisper ASR.
///
/// Run with: cargo run --example audio_captcha --features audio

#[cfg(feature = "audio")]
use anyhow::{Context, Result};
#[cfg(feature = "audio")]
use argus_captcha::{CaptchaSolver, Solution};
#[cfg(feature = "audio")]
use std::fs;
#[cfg(feature = "audio")]
use std::path::Path;

#[cfg(feature = "audio")]
#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("=== Audio CAPTCHA Solver Example ===\n");

    // 1. Create solver with audio enabled
    let mut solver = CaptchaSolver::new()?;
    println!("✓ Solver initialized with Whisper support\n");

    // 2. Solve from WAV file
    println!("--- Example 1: WAV File ---");
    if Path::new("examples/data/captcha_audio.wav").exists() {
        solve_audio_file(&mut solver, "examples/data/captcha_audio.wav").await?;
    } else {
        println!("  ℹ Sample file not found (create examples/data/captcha_audio.wav)");
        demonstrate_audio_solving(&mut solver).await?;
    }

    // 3. Solve from MP3 file
    println!("\n--- Example 2: MP3 File ---");
    if Path::new("examples/data/captcha_audio.mp3").exists() {
        solve_audio_file(&mut solver, "examples/data/captcha_audio.mp3").await?;
    } else {
        println!("  ℹ Sample file not found (create examples/data/captcha_audio.mp3)");
    }

    // 4. Solve from downloaded reCAPTCHA audio
    println!("\n--- Example 3: reCAPTCHA Audio ---");
    demonstrate_recaptcha_audio(&mut solver).await?;

    // 5. Print metrics
    println!("\n=== Performance Metrics ===");
    println!("Success rate: {:.2}%", solver.success_rate() * 100.0);
    println!("Avg solve time: {:?}", solver.avg_solve_time());

    Ok(())
}

#[cfg(feature = "audio")]
async fn solve_audio_file(solver: &mut CaptchaSolver, path: &str) -> Result<()> {
    println!("  Loading audio from: {}", path);

    // Read audio file
    let audio_bytes = fs::read(path).context("Failed to read audio file")?;

    println!("  File size: {} bytes", audio_bytes.len());
    println!("  Transcribing with Whisper...");

    // Solve
    let start = std::time::Instant::now();
    let result = solver.solve_audio(&audio_bytes).await?;
    let elapsed = start.elapsed();

    // Print result
    if let Solution::AudioText(text) = &result.solution {
        println!("  ✓ Transcription: \"{}\"", text);
        println!("  Confidence: {:.2}%", result.confidence * 100.0);
        println!("  Solve time: {:?}", elapsed);

        solver.update_metrics(&result, true);
    }

    Ok(())
}

#[cfg(feature = "audio")]
async fn demonstrate_audio_solving(solver: &mut CaptchaSolver) -> Result<()> {
    println!("  Demonstrating with synthetic audio...\n");

    // Create a simple WAV file in memory (440Hz sine wave - "A" note)
    // Real audio would be much more complex
    let sample_rate = 16000;
    let duration_secs = 3;
    let samples: Vec<f32> = (0..sample_rate * duration_secs)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
        })
        .collect();

    // Convert to WAV bytes
    let wav_bytes = create_wav_bytes(&samples, sample_rate)?;

    println!(
        "  Created synthetic audio ({} samples, {}Hz)",
        samples.len(),
        sample_rate
    );
    println!("  Note: Real CAPTCHAs contain speech with numbers/letters\n");

    // Solve (will likely return garbage for synthetic audio)
    match solver.solve_audio(&wav_bytes).await {
        Ok(result) => {
            if let Solution::AudioText(text) = &result.solution {
                println!("  Result: \"{}\"", text);
                println!("  (This is expected to be incorrect for synthetic audio)");
            }
        }
        Err(e) => {
            println!("  Expected error with synthetic audio: {}", e);
        }
    }

    Ok(())
}

#[cfg(feature = "audio")]
async fn demonstrate_recaptcha_audio(solver: &mut CaptchaSolver) -> Result<()> {
    println!("  Simulating reCAPTCHA audio challenge...\n");

    println!("  In a real scenario:");
    println!("  1. Click reCAPTCHA checkbox");
    println!("  2. Click audio challenge button");
    println!("  3. Wait for audio to load");
    println!("  4. Download audio file from:");
    println!("     document.querySelector('.rc-audiochallenge-tdownload-link').href");
    println!("  5. Pass audio bytes to solver.solve_audio()");
    println!("  6. Enter transcription and submit\n");

    println!("  Example code:");
    println!(
        r#"
    // Download audio
    let audio_url = page.evaluate(
        "document.querySelector('.rc-audiochallenge-tdownload-link').href"
    ).await?.into_value::<String>()?;

    let audio_bytes = reqwest::get(&audio_url)
        .await?
        .bytes()
        .await?;

    // Solve
    let result = solver.solve_audio(&audio_bytes).await?;

    if let Solution::AudioText(text) = &result.solution {{
        // Enter solution
        page.evaluate(&format!(
            "document.querySelector('#audio-response').value = '{{}}'",
            text
        )).await?;

        // Submit
        page.evaluate(
            "document.querySelector('#recaptcha-verify-button').click()"
        ).await?;
    }}
    "#
    );

    Ok(())
}

#[cfg(feature = "audio")]
fn create_wav_bytes(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    use hound::{WavSpec, WavWriter};

    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut cursor, spec)?;
        for &sample in samples {
            let amplitude = (sample * i16::MAX as f32) as i16;
            writer.write_sample(amplitude)?;
        }
        writer.finalize()?;
    }

    Ok(cursor.into_inner())
}

#[cfg(not(feature = "audio"))]
fn main() {
    eprintln!("This example requires the 'audio' feature.");
    eprintln!("Run with: cargo run --example audio_captcha --features audio");
    std::process::exit(1);
}
