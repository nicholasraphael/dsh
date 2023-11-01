mod builtins;
mod internals;
mod utils;

use ai_engine::AIEngine;
use internals::commands;

use crossterm::{
    style::{Color, Print, ResetColor, SetForegroundColor},
    ExecutableCommand,
};
use std::{
    io::{self, Write},
    process,
};

use utils::setup_workdir;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Setup a Ctrl+C handler to ignore the signal
    ctrlc::set_handler(|| {}).expect("Error setting Ctrl+C handler");

    //  let mut error_output_map: HashMap<u32, String> = HashMap::new();
    let mut ai_engine = AIEngine::default()?;

    let mut workdir = setup_workdir();
    loop {
        workdir = setup_workdir();

        std::io::stdout()
            .execute(SetForegroundColor(Color::Blue))?
            .execute(Print(workdir?.as_str()))?
            .execute(ResetColor)?;
        io::stdout().flush().expect("Failed to flush stdout");

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        let args: Vec<&str> = input.split_whitespace().collect();
        match args[0] {
            "cd" => {
                builtins::cd::run(input);
            }
            "help" => {
                todo!();
            }
            "exit" => process::exit(0),
            _ => {
                let commands: Vec<&str> = input.split('|').map(|s| s.trim()).collect();

                if commands.len() > 1 {
                    // Handle pipes
                    todo!()
                } else {
                    // Handle single commands
                    commands::run_single_command(input, &mut ai_engine).await;
                }
            }
        }
    }
}
