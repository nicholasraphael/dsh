use anyhow::Result;
use candle_core::Device;
use tokenizers::Tokenizer;

pub fn format_size(size_in_bytes: usize) -> String {
  if size_in_bytes < 1_000 {
      format!("{}B", size_in_bytes)
  } else if size_in_bytes < 1_000_000 {
      format!("{:.2}KB", size_in_bytes as f64 / 1e3)
  } else if size_in_bytes < 1_000_000_000 {
      format!("{:.2}MB", size_in_bytes as f64 / 1e6)
  } else {
      format!("{:.2}GB", size_in_bytes as f64 / 1e9)
  }
}

pub fn device(cpu: bool) -> Result<Device> {
  if cpu {
      Ok(Device::Cpu)
  } else {
      let device = Device::cuda_if_available(0)?;
      if !device.is_cuda() {
          println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
      }
      Ok(device)
  }
}


pub fn print_token(next_token: u32, tokenizer: &Tokenizer) {
  // Extracting the last token as a string is complicated, here we just apply some simple
  // heuristics as it seems to work well enough for this example. See the following for more
  // details:
  // https://github.com/huggingface/tokenizers/issues/1141#issuecomment-1562644141
  if let Some(text) = tokenizer.id_to_token(next_token) {
      let text = text.replace('▁', " ");
      let ascii = text
          .strip_prefix("<0x")
          .and_then(|t| t.strip_suffix('>'))
          .and_then(|t| u8::from_str_radix(t, 16).ok());
      match ascii {
          None => print!("{text}"),
          Some(ascii) => {
              if let Some(chr) = char::from_u32(ascii as u32) {
                  if chr.is_ascii() {
                      print!("{chr}")
                  }
              }
          }
      }
      let _ = std::io::Write::flush(&mut std::io::stdout());
  }
}