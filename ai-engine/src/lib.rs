mod args;
mod utils;

use std::io::Write;

use anyhow::{Error, Result, Ok};
use args::{Args, Prompt, Which};
use utils::{format_size, print_token};

use candle_core::quantized::{ggml_file, gguf_file};
use candle_transformers::models::quantized_llama::{self as model, ModelWeights};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

pub struct AIEngine {
    model: ModelWeights,
    args: Args
}

impl AIEngine {
    pub fn default() -> Result<Self, Error> {
        let model_args = Args::default();
        return AIEngine::new(&model_args);
    }

    pub fn new(model_args: &Args) -> Result<Self, Error> {
        let model_path = model_args.model()?;
        let mut file = std::fs::File::open(&model_path).unwrap();

        let mut model = match model_path.extension().and_then(|v| v.to_str()) {
            Some("gguf") => {
                let model = gguf_file::Content::read(&mut file)?;
                let mut total_size_in_bytes = 0;
                for (_, tensor) in model.tensor_infos.iter() {
                    let elem_count = tensor.shape.elem_count();
                    total_size_in_bytes +=
                        elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.blck_size();
                }
                println!(
                    "loaded {:?} tensors ({})",
                    model.tensor_infos.len(),
                    &format_size(total_size_in_bytes)
                );
                model::ModelWeights::from_gguf(model, &mut file)?
            }
            Some("ggml" | "bin") | Some(_) | None => {
                let model = ggml_file::Content::read(&mut file)?;
                let mut total_size_in_bytes = 0;
                for (_, tensor) in model.tensors.iter() {
                    let elem_count = tensor.shape().elem_count();
                    total_size_in_bytes +=
                        elem_count * tensor.dtype().type_size() / tensor.dtype().blck_size();
                }
                println!(
                    "loaded {:?} tensors ({})",
                    model.tensors.len(),
                    &format_size(total_size_in_bytes)
                );
                println!("params: {:?}", model.hparams);
                let default_gqa = match model_args.which {
                    Which::L7b
                    | Which::L13b
                    | Which::L7bChat
                    | Which::L13bChat
                    | Which::L7bCode
                    | Which::L13bCode
                    | Which::L34bCode => 1,
                    Which::Mistral7b | Which::Mistral7bInstruct | Which::L70b | Which::L70bChat => {
                        8
                    }
                };
                model::ModelWeights::from_ggml(model, model_args.gqa.unwrap_or(default_gqa))?
            }
        };
        println!("model built");
        return Ok(AIEngine { model, args: model_args.clone() });
    }

    pub fn inference(&mut self, input: &str) -> Result<()> {
        self.args.prompt = Some(input.to_string());
        let prompt = match self.args.prompt.as_deref() {
            Some(s) => Prompt::One(s.to_string()),
            None => Prompt::One("Say you are a smart model in the computer shell".to_string())
        };

        let mut pre_prompt_tokens = vec![];

        let prompt_str = match &prompt {
            Prompt::One(prompt) => prompt.clone(),
            Prompt::Interactive => todo!(),
            Prompt::Chat => todo!(),
        };
        let tokenizer = self.args.tokenizer().unwrap();
        let tokens = tokenizer
            .encode(prompt_str, true)
            .map_err(anyhow::Error::msg)
            .unwrap();

        let prompt_tokens = [&pre_prompt_tokens, tokens.get_ids()].concat();
        loop {
            let prompt_str = match &prompt {
                Prompt::One(prompt) => prompt.clone(),
                Prompt::Interactive | Prompt::Chat => {
                    print!("> ");
                    std::io::stdout().flush()?;
                    let mut prompt = String::new();
                    std::io::stdin().read_line(&mut prompt)?;
                    if prompt.ends_with('\n') {
                        prompt.pop();
                        if prompt.ends_with('\r') {
                            prompt.pop();
                        }
                    }
                    if self.args.which.is_mistral() {
                        format!("[INST] {prompt} [/INST]")
                    } else {
                        prompt
                    }
                }
            };
            print!("{}", &prompt_str);
            let tokens = tokenizer
                .encode(prompt_str, true)
                .map_err(anyhow::Error::msg)?;
            if self.args.verbose_prompt {
                for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                    let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                    println!("{id:7} -> '{token}'");
                }
            }

            let prompt_tokens = [&pre_prompt_tokens, tokens.get_ids()].concat();
            let to_sample = self.args.sample_len.saturating_sub(1);
            let prompt_tokens = if prompt_tokens.len() + to_sample > model::MAX_SEQ_LEN - 10 {
                let to_remove = prompt_tokens.len() + to_sample + 10 - model::MAX_SEQ_LEN;
                prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
            } else {
                prompt_tokens
            };
            let mut all_tokens = vec![];
            let temperature = Some(self.args.temperature);
            let mut logits_processor = LogitsProcessor::new(self.args.seed, temperature, self.args.top_p);

            let start_prompt_processing = std::time::Instant::now();
            let mut next_token = {
                let input = Tensor::new(prompt_tokens.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
                let logits = self.model.forward(&input, 0)?;
                let logits = logits.squeeze(0)?;
                logits_processor.sample(&logits)?
            };
            let prompt_dt = start_prompt_processing.elapsed();
            all_tokens.push(next_token);
            print_token(next_token, &tokenizer);

            let eos_token = *tokenizer.get_vocab(true).get("</s>").unwrap();

            let start_post_prompt = std::time::Instant::now();
            for index in 0..to_sample {
                let input = Tensor::new(&[next_token], &Device::Cpu)?.unsqueeze(0)?;
                let logits = self.model.forward(&input, prompt_tokens.len() + index)?;
                let logits = logits.squeeze(0)?;
                let logits = if self.args.repeat_penalty == 1. {
                    logits
                } else {
                    let start_at = all_tokens.len().saturating_sub(self.args.repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        self.args.repeat_penalty,
                        &all_tokens[start_at..],
                    )?
                };
                next_token = logits_processor.sample(&logits)?;
                all_tokens.push(next_token);
                print_token(next_token, &tokenizer);
                if next_token == eos_token {
                    break;
                };
            }
            let dt = start_post_prompt.elapsed();
            println!(
                "\n\n{:4} prompt tokens processed: {:.2} token/s",
                prompt_tokens.len(),
                prompt_tokens.len() as f64 / prompt_dt.as_secs_f64(),
            );
            println!(
                "{:4} tokens generated: {:.2} token/s",
                to_sample,
                to_sample as f64 / dt.as_secs_f64(),
            );

            match prompt {
                Prompt::One(_) => break,
                Prompt::Interactive => {}
                Prompt::Chat => {
                    pre_prompt_tokens = [prompt_tokens.as_slice(), all_tokens.as_slice()].concat()
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_inference() {
        let mut ai_engine = AIEngine::default().unwrap();
        let _ = ai_engine.inference("write a add function in rust");
    }
}
