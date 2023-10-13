
use std::env;
use whoami;

pub fn setup_workdir() -> Result<String, std::io::Error> {
    let binding = env::current_dir()?;
    let workdir = binding.to_string_lossy();
    let user = format!("{} {}", whoami::username(), "@ ");

    let final_workdir = format!("{}{} dsh % ", user, workdir);
    Ok(final_workdir)
}
