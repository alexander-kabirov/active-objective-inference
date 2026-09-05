use std::{
    fs,
    path::PathBuf,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;

const GROQ_URL: &str = "https://api.groq.com/openai/v1/chat/completions";
const OPENAI_URL: &str = "https://api.openai.com/v1/responses";
const DEFAULT_MODEL: &str = "openai/gpt-oss-120b";
const DEFAULT_OPENAI_MODEL: &str = "gpt-5.4-mini";

#[derive(Clone, Copy)]
enum Provider {
    Groq,
    OpenAi,
}

#[derive(Clone, Serialize)]
pub struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    messages: &'a [Message],
    reasoning_effort: &'a str,
    include_reasoning: bool,
    max_completion_tokens: u32,
}

#[derive(Deserialize)]
struct Response {
    choices: Vec<Choice>,
    usage: Option<Value>,
}

#[derive(Deserialize)]
struct Choice {
    message: Answer,
    finish_reason: Option<String>,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Answer {
    pub content: Option<String>,
    pub reasoning: Option<String>,
    #[serde(skip_deserializing, default)]
    pub finish_reason: Option<String>,
    #[serde(skip_deserializing, default)]
    pub usage: Option<Value>,
    #[serde(skip_deserializing, default)]
    pub raw_response: Option<Value>,
    #[serde(skip_deserializing, default)]
    pub format_attempt: usize,
    #[serde(skip_deserializing, default)]
    pub discarded_format_responses: Vec<Value>,
}

#[derive(Clone)]
pub struct ModelClient {
    http: reqwest::Client,
    api_key: String,
    model: String,
    reasoning_effort_override: Option<String>,
    provider: Provider,
}

impl ModelClient {
    pub fn from_env() -> Result<Self> {
        let provider = match std::env::var("MODEL_PROVIDER").as_deref() {
            Ok("openai") => Provider::OpenAi,
            Ok("groq") | Err(_) => Provider::Groq,
            Ok(value) => bail!("unsupported MODEL_PROVIDER {value:?}"),
        };
        let (api_key, model) = match provider {
            Provider::Groq => (
                std::env::var("GROQ_API_KEY").context("GROQ_API_KEY is not set")?,
                std::env::var("GROQ_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_owned()),
            ),
            Provider::OpenAi => (
                std::env::var("OPENAI_API_KEY").context("OPENAI_API_KEY is not set")?,
                std::env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_OPENAI_MODEL.to_owned()),
            ),
        };
        let reasoning_effort_override = match provider {
            Provider::Groq => std::env::var("GROQ_REASONING_EFFORT_OVERRIDE").ok(),
            Provider::OpenAi => std::env::var("OPENAI_REASONING_EFFORT_OVERRIDE").ok(),
        };
        Ok(Self {
            http: reqwest::Client::new(),
            api_key,
            model,
            reasoning_effort_override,
            provider,
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn reasoning_effort_override(&self) -> Option<&str> {
        self.reasoning_effort_override.as_deref()
    }

    pub fn provider_name(&self) -> &'static str {
        match self.provider {
            Provider::Groq => "groq",
            Provider::OpenAi => "openai",
        }
    }

    pub fn endpoint(&self) -> &'static str {
        match self.provider {
            Provider::Groq => GROQ_URL,
            Provider::OpenAi => OPENAI_URL,
        }
    }

    pub fn returns_observable_reasoning(&self) -> bool {
        matches!(self.provider, Provider::Groq)
    }

    pub async fn call_json<T: DeserializeOwned>(
        &self,
        messages: &[Message],
        effort: &str,
        tokens: u32,
        stage: &str,
    ) -> Result<(Answer, T)> {
        const MAX_FORMAT_ATTEMPTS: usize = 6;
        let mut discarded = Vec::new();
        for format_attempt in 1..=MAX_FORMAT_ATTEMPTS {
            let mut answer = self.call(messages, effort, tokens).await?;
            answer.format_attempt = format_attempt;
            if let Some(content) = answer.content.as_deref()
                && let Ok(parsed) = parse_json(content)
            {
                answer.discarded_format_responses = discarded;
                return Ok((answer, parsed));
            }
            discarded.push(serde_json::to_value(&answer)?);
            eprintln!(
                "{stage} returned invalid JSON; format retry {format_attempt}/{MAX_FORMAT_ATTEMPTS}"
            );
        }
        bail!("{stage} failed to return valid JSON after {MAX_FORMAT_ATTEMPTS} format attempts")
    }

    async fn call(&self, messages: &[Message], effort: &str, tokens: u32) -> Result<Answer> {
        match self.provider {
            Provider::Groq => self.call_groq(messages, effort, tokens).await,
            Provider::OpenAi => self.call_openai(messages, effort, tokens).await,
        }
    }

    async fn call_groq(&self, messages: &[Message], effort: &str, tokens: u32) -> Result<Answer> {
        let effort = self.reasoning_effort_override.as_deref().unwrap_or(effort);
        let request = Request {
            model: &self.model,
            messages,
            reasoning_effort: effort,
            include_reasoning: true,
            max_completion_tokens: tokens,
        };

        for attempt in 1..=6 {
            let response = match self
                .http
                .post(GROQ_URL)
                .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
                .header(CONTENT_TYPE, "application/json")
                .json(&request)
                .send()
                .await
            {
                Ok(response) => response,
                Err(error) if attempt < 6 => {
                    eprintln!("Transient Groq transport error ({error}); retrying in 8s...");
                    tokio::time::sleep(Duration::from_secs(8)).await;
                    continue;
                }
                Err(error) => return Err(error).context("failed to call Groq"),
            };
            let status = response.status();
            let retry = response
                .headers()
                .get("retry-after")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(15)
                .clamp(8, 60);
            let body = response.text().await?;
            if status == reqwest::StatusCode::TOO_MANY_REQUESTS && attempt < 6 {
                eprintln!("Rate limited; retrying in {retry}s...");
                tokio::time::sleep(Duration::from_secs(retry)).await;
                continue;
            }
            if !status.is_success() {
                bail!("Groq returned {status}: {body}")
            }
            let raw_response: Value = serde_json::from_str(&body)?;
            let decoded: Response = serde_json::from_value(raw_response.clone())?;
            let choice = decoded
                .choices
                .first()
                .context("Groq returned no choices")?;
            let mut answer = choice.message.clone();
            answer.finish_reason = choice.finish_reason.clone();
            answer.usage = decoded.usage;
            answer.raw_response = Some(raw_response);
            return Ok(answer);
        }
        unreachable!()
    }

    async fn call_openai(&self, messages: &[Message], effort: &str, tokens: u32) -> Result<Answer> {
        let effort = self.reasoning_effort_override.as_deref().unwrap_or(effort);
        let request = serde_json::json!({
            "model": self.model,
            "input": messages,
            "store": false,
            "max_output_tokens": tokens,
            "reasoning": {"effort": effort},
            "text": {
                "verbosity": "low",
                "format": {"type": "json_object"}
            }
        });
        for attempt in 1..=6 {
            let response = match self
                .http
                .post(OPENAI_URL)
                .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
                .header(CONTENT_TYPE, "application/json")
                .json(&request)
                .send()
                .await
            {
                Ok(response) => response,
                Err(error) if attempt < 6 => {
                    eprintln!("Transient OpenAI transport error ({error}); retrying in 8s...");
                    tokio::time::sleep(Duration::from_secs(8)).await;
                    continue;
                }
                Err(error) => return Err(error).context("failed to call OpenAI"),
            };
            let status = response.status();
            let retry = response
                .headers()
                .get("retry-after")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(15)
                .clamp(8, 60);
            let body = response.text().await?;
            if status == reqwest::StatusCode::TOO_MANY_REQUESTS && attempt < 6 {
                eprintln!("OpenAI rate limited; retrying in {retry}s...");
                tokio::time::sleep(Duration::from_secs(retry)).await;
                continue;
            }
            if !status.is_success() {
                bail!("OpenAI returned {status}: {body}")
            }
            let raw_response: Value = serde_json::from_str(&body)?;
            let content = raw_response
                .get("output")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|item| item.get("content").and_then(Value::as_array))
                .flatten()
                .find_map(|item| {
                    (item.get("type").and_then(Value::as_str) == Some("output_text"))
                        .then(|| item.get("text").and_then(Value::as_str))
                        .flatten()
                        .map(str::to_owned)
                });
            let response_status = raw_response
                .get("status")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let usage = raw_response.get("usage").cloned();
            return Ok(Answer {
                content,
                reasoning: None,
                finish_reason: response_status,
                usage,
                raw_response: Some(raw_response),
                format_attempt: 0,
                discarded_format_responses: Vec::new(),
            });
        }
        unreachable!()
    }
}

pub fn user(content: &str) -> Message {
    Message {
        role: "user".to_owned(),
        content: content.to_owned(),
    }
}

pub fn developer(content: &str) -> Message {
    Message {
        role: "developer".to_owned(),
        content: content.to_owned(),
    }
}

pub fn assistant(content: &str) -> Message {
    Message {
        role: "assistant".to_owned(),
        content: content.to_owned(),
    }
}

pub fn print_stage(name: &str, answer: &Answer) {
    println!(
        "\n=== {name}: REASONING ===\n{}\n\n=== {name}: FINAL ANSWER ===\n{}",
        answer.reasoning.as_deref().unwrap_or("<not returned>"),
        answer.content.as_deref().unwrap_or("<not returned>")
    );
}

pub fn result_path(prefix: &str) -> Result<PathBuf> {
    fs::create_dir_all("results")?;
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    Ok(PathBuf::from(format!("results/{prefix}-{stamp}.jsonl")))
}

fn parse_json<T: DeserializeOwned>(text: &str) -> Result<T> {
    let start = text.find('{').context("response contained no JSON")?;
    let end = text.rfind('}').context("response contained no JSON")?;
    serde_json::from_str(&text[start..=end]).context("invalid JSON")
}
