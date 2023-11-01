use core::time;
use std::process::{ChildStdout, Stdio};
use tokio::process::Command;
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    sync::mpsc,
};

use ai_engine::AIEngine;
use anyhow::{Error, Result};
use regex::Regex;

enum Message {
    Data(String),
    Done,
}

pub async fn run_single_command(command: &str, engine: &mut AIEngine) -> Result<(), Error> {
    let commands: Vec<&str> = command.split_whitespace().map(|c| c.trim()).collect();
    let program = commands[0];
    let args = &commands[1..];

    let mut child = Command::new(program)
        .args(args)
        .stderr(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;

    let stdout = child.stdout.take().expect("Error getting stdout");
    let stderr = child.stderr.take().unwrap();

    let mut reader = BufReader::new(stdout).lines();
    let mut err_reader = BufReader::new(stderr).lines();

    let (mut tx, mut rx) = mpsc::channel::<String>(32);

    // Ensure the child process is spawned in the runtime so it can
    // make progress on its own while we await for any output.
    tokio::spawn(async move {
        let _ = child
            .wait()
            .await
            .expect("child process encountered an error");
    });

    let mut prompt = String::new();

    let error_re = Regex::new(r"(?i)error")?;

    tokio::spawn(async move {
        while let Ok(Some(line)) = reader.next_line().await {
            println!("{}", line);
            if error_re.is_match(&line) {
                if let Err(_) = tx.send(line).await {
                    println!("receiver dropped");
                }
            }
        }
    });


    while let Some(msg) = rx.recv().await {
        prompt.push_str(&msg);

    }

    if let Err(err) = engine.inference_openai(&prompt).await {
        println!("error with generating a fix. {:?}", err)
    }

   

    Ok(())
}

pub fn run_piped_commands() {}
