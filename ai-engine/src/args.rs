use tokenizers::Tokenizer;

#[derive(Debug)]
pub enum Prompt {
    Interactive,
    Chat,
    One(String),
}

#[derive(Clone, Debug, Copy)]
pub enum Which {
    L7b,
    L13b,
    L70b,
    L7bChat,
    L13bChat,
    L70bChat,
    L7bCode,
    L13bCode,
    L34bCode,
    Mistral7b,
    Mistral7bInstruct,
    RiftSolver
}

impl Which {
    pub fn is_mistral(&self) -> bool {
        match self {
            Self::L7b
            | Self::L13b
            | Self::L70b
            | Self::L7bChat
            | Self::L13bChat
            | Self::L70bChat
            | Self::L7bCode
            | Self::L13bCode
            | Self::L34bCode 
            | Self::RiftSolver => false,
            Self::Mistral7b | Self::Mistral7bInstruct => true,
        }
    }
}

#[derive(Clone)]
pub struct Args {
  /// GGML file to load, typically a .bin file generated by the quantize command from llama.cpp
  pub model: Option<String>,

  /// The initial prompt, use 'interactive' for entering multiple prompts in an interactive way
  /// and 'chat' for an interactive model where history of previous prompts and generated tokens
  /// is preserved.
  pub prompt: Option<String>,

  /// The length of the sample to generate (in tokens).
  pub sample_len: usize,

  /// The tokenizer config in json format.
  pub tokenizer: Option<String>,

  /// The temperature used to generate samples, use 0 for greedy sampling.
  pub temperature: f64,

  /// Nucleus sampling probability cutoff.
  pub top_p: Option<f64>,

  /// The seed to use when generating random samples.
  pub seed: u64,

  /// Enable tracing (generates a trace-timestamp.json file).
  pub tracing: bool,

  /// Display the token for the specified prompt.
  pub verbose_prompt: bool,

  /// Penalty to be applied for repeating tokens, 1. means no penalty.
  pub repeat_penalty: f32,

  /// The context size to consider for the repeat penalty.
  pub repeat_last_n: usize,

  /// The model size to use.
  pub which: Which,

  /// Group-Query Attention, use 8 for the 70B version of LLaMAv2.
  pub gqa: Option<usize>,
}

impl Default for Args {
  fn default() -> Args {
      Args {
          model: None,
          prompt: None,
          sample_len: 1500,
          tokenizer: None,
          temperature: 1.0, 
          top_p: None,
          seed: 299792458,
          tracing: false,
          verbose_prompt: false,
          repeat_penalty: 1.1,
          repeat_last_n: 64,
          gqa: None,
          which: Which::L7bChat,
      }
  }
}

impl Args {
  pub fn tokenizer(&self) -> anyhow::Result<Tokenizer> {
      let tokenizer_path = match &self.tokenizer {
          Some(config) => std::path::PathBuf::from(config),
          None => {
              let api = hf_hub::api::sync::Api::new()?;
              let repo = if self.which.is_mistral() {
                  "mistralai/Mistral-7B-v0.1"
              } else {
                  "hf-internal-testing/llama-tokenizer"
              };
              let api = api.model(repo.to_string());
              api.get("tokenizer.json")?
          }
      };
      Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)
  }

  pub fn model(&self) -> anyhow::Result<std::path::PathBuf> {
      let model_path = match &self.model {
          Some(config) => std::path::PathBuf::from(config),
          None => {
              let (repo, filename) = match self.which {
                  Which::L7b => ("TheBloke/Llama-2-7B-GGML", "llama-2-7b.ggmlv3.q4_0.bin"),
                  Which::L13b => ("TheBloke/Llama-2-13B-GGML", "llama-2-13b.ggmlv3.q4_0.bin"),
                  Which::L70b => ("TheBloke/Llama-2-70B-GGML", "llama-2-70b.ggmlv3.q4_0.bin"),
                  Which::L7bChat => (
                      "TheBloke/Llama-2-7B-Chat-GGML",
                      "llama-2-7b-chat.ggmlv3.q4_0.bin",
                  ),
                  Which::L13bChat => (
                      "TheBloke/Llama-2-13B-Chat-GGML",
                      "llama-2-13b-chat.ggmlv3.q4_0.bin",
                  ),
                  Which::L70bChat => (
                      "TheBloke/Llama-2-70B-Chat-GGML",
                      "llama-2-70b-chat.ggmlv3.q4_0.bin",
                  ),
                  Which::L7bCode => ("TheBloke/CodeLlama-7B-GGUF", "codellama-7b.Q8_0.gguf"),
                  Which::L13bCode => ("TheBloke/CodeLlama-13B-GGUF", "codellama-13b.Q8_0.gguf"),
                  Which::L34bCode => ("TheBloke/CodeLlama-34B-GGUF", "codellama-34b.Q8_0.gguf"),
                  Which::Mistral7b => (
                      "TheBloke/Mistral-7B-v0.1-GGUF",
                      "mistral-7b-v0.1.Q4_K_S.gguf",
                  ),
                  Which::Mistral7bInstruct => (
                      "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                      "mistral-7b-instruct-v0.1.Q4_K_S.gguf",
                  ),
                  Which::RiftSolver => (
                    "morph-labs/morph-prover-v0-7b-gguf",
                    "gguf-model-Q8_0.gguf"
                  )
              };
              let api = hf_hub::api::sync::Api::new()?;
              let api = api.model(repo.to_string());
              api.get(filename)?
          }
      };
      Ok(model_path)
  }
}