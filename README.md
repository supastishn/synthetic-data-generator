# Prompt and Answer Generation Tool

This tool generates structured conversations (prompts and answers) based on specified topics using language models.

## Requirements
- Requires LiteLLM for API access

## Features
- Generate multiple prompts per topic
- Generate assistant answers for each prompt
- Save conversations to JSON file
- Environment variables for configuration
- Interactive prompts for missing values
- Batch processing for prompt/answer generation
- Asynchronous API calls for improved performance

## Setup
1. Install dependencies:
```bash
pip install python-dotenv litellm
```
2. Create a `.env` file (optional) with your environment variables:
```ini
PROMPTGEN_MODEL=<your_prompt_generation_model>
ANSWERGEN_MODEL=<your_answer_generation_model>[,<other_models>]
TEMPERATURE=0.7
TOPICS=topic1,topic2
AMOUNTS=3,2
MULTI_PROMPT=y
LOGITS=y    # Set to y to enable logits capture
OUTPUT_FILE=my_conversations.json
MODEL_SPLIT=50,25,25   # For multiple answer models, comma-separated percentages (sum=100)
```

## API Keys
Set API keys using environment variables before running:
```bash
export OPENAI_API_KEY=sk-xxxx  # For OpenAI models
export ANTHROPIC_API_KEY=sk-xxxx  # For Claude models
```
The key required depends on your preferred provider:
- Follow LiteLLM's environment variable naming: https://litellm.vercel.app/docs/providers
- Keys should be set in your shell or `.env` file
- Run `litellm --help` to see all supported providers

## Logits Capture
When `LOGITS=y`:
- Assistant responses will include token-level probability data from the model
- This data includes the top 10 token candidates at each position with their log probabilities

## Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `PROMPTGEN_MODEL` | Model for prompt generation | *Required* |
| `ANSWERGEN_MODEL` | Model for answer generation | *Required* |
| `TEMPERATURE` | Creativity level (0.0-1.0) | 0.7 |
| `TOPICS` | Comma-separated list of topics | *Required* |
| `AMOUNTS` | Number of prompts per topic (single or comma-separated) | *Required* |
| `MULTI_PROMPT` | Use multi-prompt generation? (Y/n) | y |
| `MODEL_SPLIT` | Percentage split for answer models (comma-separated, sum=100) | Required for multiple models |
| `LOGITS` | Use logits for answer generation? (y/n) | n |
| `OUTPUT_FILE` | Output JSON filename | conversations.json |
| `BATCH_SIZE` | Batch size for prompt generation | 5 |
| `ASYNC_GEN` | Enable asynchronous generation? (y/n) | n |
| `VERBOSE_LOGGING` | Print request/response bodies | n |

## Usage
Run the script:
```bash
python main.py
```

The tool will:
1. Check for required environment variables
2. Prompt for missing values
3. Generate prompts for each topic
4. Generate answers for each prompt
5. Save conversations to specified JSON file

## Output Format
Conversations are saved in JSON format:
```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "Explain quantum computing in simple terms",
        "generation_model": "gpt-3.5-turbo"  // New field for prompt model
      },
      {
        "role": "assistant",
        "content": "Quantum computing leverages quantum mechanics to process information...",
        "logprobs": {
          "content": [
            {
              "token": "Quantum",
              "logprob": -0.1,
              "top_logprobs": [
                {"token": "Quantum", "logprob": -0.1},
                {"token": "This", "logprob": -1.2},
                ...
              ]
            }
          ]
        },
        "generation_model": "gpt-4"  // New field for answer model
      }
    ],
    "model": "gpt-4"  // Model used for answer generation in this conversation
  }
]
```
- The conversation object now includes a top-level "model" field indicating the answer generation model
- User messages include "generation_model" showing which model created the prompt

## Example
```bash
# .env file:
PROMPTGEN_MODEL=gpt-3.5-turbo
ANSWERGEN_MODEL=gpt-4
TOPICS=Python,JavaScript
AMOUNTS=2
BATCH_SIZE=5          # Add this line
ASYNC_GEN=n           # Add this line

# Command:
python main.py
```

## Notes
- Uses [LiteLLM](https://github.com/BerriAI/liteLLM) format for API access
- Check `.env.example` for configuration reference
```
