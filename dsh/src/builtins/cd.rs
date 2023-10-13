use std::env;
use anyhow::Error;

pub fn run(args: &str) ->Result<(), Error> {
  let args_split: Vec<&str> = args.split_whitespace().collect();
  let args_len = args_split.len();
  
  if args_len > 2 {
    println!("dsh: cd: too many arguments");
  }
 
  if (args_split.len() < 2) || (args_split[1].is_empty()) || (args_split[1] == "~") {
    let home = std::env::var("HOME")?;
    if let Err(err) = env::set_current_dir(home) {
      println!("lsh: {}", err);
    }
  } else {
    if let Err(err) = env::set_current_dir(args_split[1]) {
        println!("lsh: {}", err);
    }
  }
  Ok(())
}