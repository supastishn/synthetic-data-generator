# Prompt and Answer Generation Tool

This tool generates structured conversations (prompts and answers) based on specified topics using language models.

## Features
- Generate multiple prompts per topic
- Generate assistant answers for each prompt
- Save conversations to JSON file
- Environment variables for configuration
- Interactive prompts for missing values

## Setup
1. Install dependencies:
```bash
pip install python-dotenv litellm
```
2. Create a `.env` file (optional) with your environment variables:
```ini
PROMPTGEN_MODEL=<your_prompt_generation_model>
ANSWERGEN_MODEL=<your_answer_generation_model>
TEMPERATURE=0.7
TOPICS=topic1,topic2
AMOUNTS=3,2
MULTI_PROMPT=y
LOGITS=n
OUTPUT_FILE=my_conversations.json
```

## Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `PROMPTGEN_MODEL` | Model for prompt generation | *Required* |
| `ANSWERGEN_MODEL` | Model for answer generation | *Required* |
| `TEMPERATURE` | Creativity level (0.0-1.0) | 0.7 |
| `TOPICS` | Comma-separated list of topics | *Required* |
| `AMOUNTS` | Number of prompts per topic (single or comma-separated) | *Required* |
| `MULTI_PROMPT` | Use multi-prompt generation? (y/n) | y |
| `LOGITS` | Use logits for answer generation? (y/n) | n |
| `OUTPUT_FILE` | Output JSON filename | conversations.json |

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
      {"role": "user", "content": "Example prompt"},
      {"role": "assistant", "content": "Example answer"}
    ]
  }
]
```

## Example
```bash
# .env file:
PROMPTGEN_MODEL=gpt-3.5-turbo
ANSWERGEN_MODEL=gpt-4
TOPICS=Python,JavaScript
AMOUNTS=2

# Command:
python main.py

# Outputs conversations to conversations.json
```
