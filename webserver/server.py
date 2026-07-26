import os
import json
import re
import subprocess
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
PROMPTGEN_MODEL = os.getenv("PROMPTGEN_MODEL", "gpt-3.5-turbo")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "conversations.json")
PENDING_FILE = os.getenv("PENDING_FILE", "pending.json")
PORT = int(os.getenv("PORT", "8000"))
def load_json_data(file_path, default_data):
    if not os.path.exists(file_path):
        return default_data
    try:
        with open(file_path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except Exception:
        return default_data
def save_json_data(file_path, data):
    with open(file_path, "w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, ensure_ascii=False, indent=2)
def set_termux_clipboard(text_content):
    try:
        process = subprocess.run(
            ["termux-clipboard-set"],
            input=text_content,
            text=True,
            capture_output=True
        )
        return process.returncode == 0
    except Exception:
        return False
def get_termux_clipboard():
    try:
        process = subprocess.run(
            ["termux-clipboard-get"],
            capture_output=True,
            text=True
        )
        if process.returncode == 0:
            return process.stdout
        return ""
    except Exception:
        return ""
def construct_prompt_creation_instruction(topic, amount, prompt_instructions):
    user_message = f"Generate exactly {amount} BRAND NEW prompts for: '{topic}'."
    if prompt_instructions:
        user_message += f"\n\nAdditional Instructions:\n{prompt_instructions}"
    user_message += """
Format requirements:
1. Each prompt must be wrapped in <prompt> tags 
2. Each prompt must contain exactly two tags: 
   - A <system> tag with a system prompt
   - A <user> tag with a user prompt
3. Output ONLY the XML-formatted prompts with no additional text
4. Example format: 
   <prompt>
      <system>System instruction for AI</system>
      <user>User question</user>
   </prompt>
5. For multiple prompts, output them consecutively without separators
"""
    return user_message.strip()
def execute_ai_prompt_generation(topic, amount, prompt_instructions, model_name=None):
    if not LITELLM_AVAILABLE:
        raise RuntimeError("LiteLLM is not installed. Please install litellm or use copy-paste mode.")
    selected_model = model_name or PROMPTGEN_MODEL
    user_message = construct_prompt_creation_instruction(topic, amount, prompt_instructions)
    response = litellm.completion(
        model=selected_model,
        messages=[
            {
                "content": "You output only in XML format. Use <prompt>, <system>, and <user> tags. Do not include any explanations or additional text.",
                "role": "system"
            },
            {"content": user_message, "role": "user"}
        ],
        temperature=1.0,
    )
    xml_content = response.choices[0].message.content
    return parse_and_add_batch_prompts(topic, xml_content, selected_model)
def add_single_manual_prompt(topic, system_prompt, user_prompt, model_name):
    current_pending_items = load_json_data(PENDING_FILE, [])
    item_id = str(len(current_pending_items) + 1)
    new_item = {
        "id": item_id,
        "topic": topic,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "status": "pending",
        "generation_model": model_name or PROMPTGEN_MODEL
    }
    current_pending_items.append(new_item)
    save_json_data(PENDING_FILE, current_pending_items)
    return [new_item]
def parse_and_add_batch_prompts(topic, raw_text, model_name):
    prompt_blocks = re.findall(r"(<prompt>.*?</prompt>)", raw_text, re.DOTALL)
    current_pending_items = load_json_data(PENDING_FILE, [])
    starting_id = len(current_pending_items) + 1
    new_items = []
    if prompt_blocks:
        for index, xml_block in enumerate(prompt_blocks):
            system_match = re.search(r"<system>(.*?)</system>", xml_block, re.DOTALL)
            user_match = re.search(r"<user>(.*?)</user>", xml_block, re.DOTALL)
            if system_match and user_match:
                item_id = str(starting_id + len(new_items))
                new_item = {
                    "id": item_id,
                    "topic": topic,
                    "system_prompt": system_match.group(1).strip(),
                    "user_prompt": user_match.group(1).strip(),
                    "status": "pending",
                    "generation_model": model_name or PROMPTGEN_MODEL
                }
                new_items.append(new_item)
                current_pending_items.append(new_item)
    if new_items:
        save_json_data(PENDING_FILE, current_pending_items)
    return new_items
HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pure Copy-Paste Synthetic Data Server</title>
    <style>
        :root {
            --bg-primary: #121212;
            --bg-secondary: #1e1e1e;
            --bg-tertiary: #2a2a2a;
            --accent: #4f46e5;
            --accent-hover: #4338ca;
            --text-primary: #e5e7eb;
            --text-secondary: #9ca3af;
            --border: #374151;
            --success: #10b981;
            --warning: #f59e0b;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            margin: 0;
            padding: 20px;
            line-height: 1.5;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        .card {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .step-header {
            color: var(--accent);
            margin-top: 0;
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
            margin-bottom: 15px;
        }
        h1, h2, h3 { margin-top: 0; color: #ffffff; }
        .form-group { margin-bottom: 15px; }
        label {
            display: block; margin-bottom: 5px; font-weight: 600; color: var(--text-secondary);
        }
        input[type="text"], input[type="number"], textarea, select {
            width: 100%; padding: 10px; background-color: var(--bg-tertiary);
            border: 1px solid var(--border); border-radius: 6px;
            color: var(--text-primary); box-sizing: border-box; font-size: 14px;
        }
        textarea { resize: vertical; min-height: 100px; }
        button {
            background-color: var(--accent); color: white; border: none;
            padding: 10px 18px; border-radius: 6px; cursor: pointer;
            font-weight: 600; font-size: 14px;
        }
        button:hover { background-color: var(--accent-hover); }
        .button-secondary { background-color: var(--bg-tertiary); border: 1px solid var(--border); }
        .button-secondary:hover { background-color: var(--border); }
        .button-success { background-color: var(--success); font-size: 15px; padding: 12px 20px; width: 100%; }
        .button-success:hover { background-color: #059669; }
        .button-group { display: flex; gap: 10px; align-items: center; margin-bottom: 15px; }
        .flex-row { display: flex; gap: 15px; }
        .flex-1 { flex: 1; }
        .prompt-box {
            background-color: var(--bg-tertiary); border-left: 4px solid var(--accent);
            padding: 12px; border-radius: 4px; white-space: pre-wrap;
            font-family: monospace; font-size: 13px; margin-bottom: 10px;
        }
        .status-badge {
            display: inline-block; padding: 4px 8px; border-radius: 4px;
            font-size: 12px; background-color: var(--accent);
        }
        .toggle-container { display: flex; align-items: center; gap: 10px; margin-bottom: 15px; }
        .tab-group {
            display: flex;
            gap: 10px;
            margin-bottom: 25px;
            border-bottom: 2px solid var(--border);
            padding-bottom: 12px;
        }
        .tab-button {
            background-color: transparent;
            color: var(--text-secondary);
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            font-weight: 600;
            font-size: 15px;
        }
        .tab-button.active {
            color: #ffffff;
            background-color: var(--bg-tertiary);
            border-radius: 6px;
        }
        .sub-tab-group {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
        }
        .sub-tab-button {
            background-color: transparent;
            color: var(--text-secondary);
            border: none;
            padding: 8px 16px;
            cursor: pointer;
        }
        .sub-tab-button.active {
            color: #ffffff;
            background-color: var(--bg-tertiary);
            border-radius: 4px;
        }
        .hidden { display: none; }
        .notification {
            padding: 10px; border-radius: 6px; background-color: var(--success);
            color: white; margin-bottom: 15px; display: none; text-align: center; font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Synthetic Data Manager</h1>
        <div id="notification" class="notification"></div>
        <div class="tab-group">
            <button id="main-tab-mode1" class="tab-button active" onclick="switchMainTab('mode1')">Mode 1: Prompt Ingestion</button>
            <button id="main-tab-mode2" class="tab-button" onclick="switchMainTab('mode2')">Mode 2: Answer Ingestion</button>
            <button id="main-tab-mode3" class="tab-button" onclick="switchMainTab('mode3')">Mode 3: Settings & Data</button>
        </div>
        <div id="panel-mode1">
            <div class="card">
                <h2>Mode 1: Prompt Generation</h2>
                <div class="sub-tab-group">
                    <button id="tab-copypaste-btn" class="sub-tab-button active" onclick="switchPromptTab('copypaste')">Copy-Paste Prompt Generator</button>
                    <button id="tab-ai-btn" class="sub-tab-button" onclick="switchPromptTab('ai')">API Prompt Generator</button>
                    <button id="tab-single-btn" class="sub-tab-button" onclick="switchPromptTab('single')">Manual Single Entry</button>
                </div>
                <div class="flex-row">
                    <div class="form-group flex-1">
                        <label for="topic">Topic</label>
                        <input type="text" id="topic" placeholder="e.g. Python Asyncio, Physics, Logic Puzzles">
                    </div>
                    <div class="form-group flex-1">
                        <label for="prompt-model-name">Prompt Generation Model Tracker</label>
                        <input type="text" id="prompt-model-name" value="gpt-4o" placeholder="e.g. gpt-4o, deepseek-coder">
                    </div>
                </div>
                <div id="copypaste-entry-panel">
                    <div class="flex-row">
                        <div class="form-group flex-1">
                            <label for="cp-amount">Amount of Prompts</label>
                            <input type="number" id="cp-amount" value="5" min="1" max="50">
                        </div>
                    </div>
                    <div class="form-group">
                        <label for="cp-instructions">Custom Instructions (Optional)</label>
                        <input type="text" id="cp-instructions" placeholder="e.g. Focus on edge cases and code examples">
                    </div>
                    <div style="margin-bottom: 20px;">
                        <button onclick="copyPromptGeneratorInstruction()" style="background-color: var(--accent); padding: 12px; font-size: 15px; width: 100%;">
                            1. Copy Prompt Creation Request to Clipboard
                        </button>
                    </div>
                    <div class="form-group">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                            <label for="ai-prompt-output-input" style="margin-bottom: 0;">2. Paste AI Output Response (XML format)</label>
                            <button class="button-secondary" onclick="pasteClipboard('ai-prompt-output-input')">Paste from Clipboard</button>
                        </div>
                        <textarea id="ai-prompt-output-input" style="min-height: 120px;" placeholder="Paste DeepSeek / Web UI XML output response here..."></textarea>
                    </div>
                    <button onclick="parseAndSaveAIPromptOutput()" style="background-color: var(--success); width: 100%; padding: 12px;">
                        3. Auto-Save Parsed Prompts to Queue
                    </button>
                </div>
                <div id="ai-entry-panel" class="hidden">
                    <div class="flex-row">
                        <div class="form-group flex-1">
                            <label for="amount">Amount</label>
                            <input type="number" id="amount" value="1" min="1" max="50">
                        </div>
                    </div>
                    <div class="form-group">
                        <label for="instructions">Custom Instructions (Optional)</label>
                        <input type="text" id="instructions" placeholder="e.g. Focus on edge cases and code examples">
                    </div>
                    <button onclick="generatePromptsAI()">API Generate & Save to Queue</button>
                </div>
                <div id="single-entry-panel" class="hidden">
                    <div class="form-group">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                            <label for="system-prompt-input" style="margin-bottom: 0;">System Prompt</label>
                            <button class="button-secondary" onclick="pasteClipboard('system-prompt-input')">Paste from Clipboard</button>
                        </div>
                        <textarea id="system-prompt-input" placeholder="Enter system prompt..."></textarea>
                    </div>
                    <div class="form-group">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                            <label for="user-prompt-input" style="margin-bottom: 0;">User Prompt</label>
                            <button class="button-secondary" onclick="pasteClipboard('user-prompt-input')">Paste from Clipboard</button>
                        </div>
                        <textarea id="user-prompt-input" placeholder="Enter user prompt..."></textarea>
                    </div>
                    <button onclick="addSinglePrompt()">Save Prompt to Queue</button>
                </div>
            </div>
        <div id="panel-mode2" class="hidden">
            <div class="card">
                <h2>Mode 2: Answer Generation (Queue Mode)</h2>
                <div id="no-pending-message">No pending prompts in queue. Create prompts in Mode 1 to start!</div>
                
                <div id="active-prompt-area" class="hidden">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span id="prompt-topic-badge" class="status-badge">Topic</span>
                        <span id="pending-counter" style="color: var(--text-secondary); font-size: 14px;">Prompt 1 of 1</span>
                    </div>

                    <div class="form-group">
                        <label for="additional-system-instructions">Additional System Instructions (Auto-injected on Copy & Submit)</label>
                        <input type="text" id="additional-system-instructions" placeholder="e.g. Speak like a pirate, include code comments, keep it concise" oninput="updateCombinedPromptPreview()">
                    </div>

                    <label>Single-Block Formatted Prompt (System + User Combined)</label>
                    <div id="combined-prompt-text" class="prompt-box"></div>

                    <div class="button-group" style="margin-bottom: 20px;">
                        <button onclick="copyFormattedPrompt()" class="button-success">
                            📋 Copy Combined Prompt Block to Clipboard
                        </button>
                    </div>
                    <hr style="border-color: var(--border); margin-bottom: 20px;">
                    <div class="form-group">
                        <label for="model-name">Answer Generation Model Name</label>
                        <input type="text" id="model-name" value="deepseek-r1-web" placeholder="e.g. deepseek-r1, gpt-4o-web">
                    </div>
                    <div class="toggle-container">
                        <input type="checkbox" id="enable-reasoning" onchange="toggleReasoningField()">
                        <label for="enable-reasoning" style="margin-bottom: 0;">Enable Reasoning Trace (&lt;think&gt; block)</label>
                    </div>
                    <div id="reasoning-group" class="form-group hidden">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                            <label for="reasoning-input" style="margin-bottom: 0;">Reasoning Trace Content</label>
                            <button class="button-secondary" onclick="pasteClipboard('reasoning-input')">Paste from Clipboard</button>
                        </div>
                        <textarea id="reasoning-input" placeholder="Paste model reasoning/thinking process here..."></textarea>
                    </div>
                    <div class="form-group">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                            <label for="response-input" style="margin-bottom: 0;">Actual AI Response</label>
                            <button class="button-secondary" onclick="pasteClipboard('response-input')">Paste from Clipboard</button>
                        </div>
                        <textarea id="response-input" placeholder="Paste final AI response here..."></textarea>
                    </div>
                    <button onclick="submitAnswer()" style="width: 100%; padding: 12px; font-size: 16px;">Submit Answer & Next in Queue</button>
                </div>
            </div>
        </div>
        <div id="panel-mode3" class="hidden">
            <div class="card">
                <h2>Global Operations</h2>
                <div class="button-group">
                    <button class="button-secondary" onclick="resetQueue('pending')" style="border-color: var(--warning); color: var(--warning);">⚠️ Clear Pending Queue</button>
                    <button class="button-secondary" onclick="resetQueue('conversations')" style="border-color: #ef4444; color: #ef4444;">🚨 Clear All Completed Conversations</button>
                </div>
            </div>
            <div class="card">
                <h2>Data Explorer</h2>
                <div class="flex-row" style="margin-bottom: 15px;">
                    <div class="form-group flex-1">
                        <label for="explorer-category">Category</label>
                        <select id="explorer-category" onchange="renderSettingsExplorer()">
                            <option value="pending">Pending Prompts</option>
                            <option value="completed">Completed Conversations</option>
                        </select>
                    </div>
                    <div class="form-group flex-1">
                        <label for="explorer-format">Format</label>
                        <select id="explorer-format" onchange="renderSettingsExplorer()">
                            <option value="human">Human-Friendly Cards</option>
                            <option value="raw">Raw JSON</option>
                        </select>
                    </div>
                </div>
                <div id="explorer-raw-container" class="hidden">
                    <button class="button-secondary" onclick="copyRawExplorerJson()" style="margin-bottom: 15px;">📋 Copy JSON to Clipboard</button>
                    <textarea id="explorer-raw-textarea" style="min-height: 350px; font-family: monospace; font-size: 13px;" readonly></textarea>
                </div>
                <div id="explorer-human-container"></div>
            </div>
        </div>
    </div>
    <script>
        let pendingPrompts = [];
        let currentIndex = 0;
        let localPending = [];
        let localConversations = [];
        function showNotification(text) {
            const el = document.getElementById("notification");
            el.innerText = text;
            el.style.display = "block";
            setTimeout(() => { el.style.display = "none"; }, 3000);
        }
        function switchMainTab(tabName) {
            document.getElementById('main-tab-mode1').classList.remove('active');
            document.getElementById('main-tab-mode2').classList.remove('active');
            document.getElementById('main-tab-mode3').classList.remove('active');
            document.getElementById('panel-mode1').classList.add('hidden');
            document.getElementById('panel-mode2').classList.add('hidden');
            document.getElementById('panel-mode3').classList.add('hidden');
            if (tabName === 'mode1') {
                document.getElementById('main-tab-mode1').classList.add('active');
                document.getElementById('panel-mode1').classList.remove('hidden');
            } else if (tabName === 'mode2') {
                document.getElementById('main-tab-mode2').classList.add('active');
                document.getElementById('panel-mode2').classList.remove('hidden');
                fetchPendingPrompts();
            } else if (tabName === 'mode3') {
                document.getElementById('main-tab-mode3').classList.add('active');
                document.getElementById('panel-mode3').classList.remove('hidden');
                loadSettingsData();
            }
        }
        function switchPromptTab(tabName) {
            document.getElementById('tab-copypaste-btn').classList.remove('active');
            document.getElementById('tab-ai-btn').classList.remove('active');
            document.getElementById('tab-single-btn').classList.remove('active');
            document.getElementById('copypaste-entry-panel').classList.add('hidden');
            document.getElementById('ai-entry-panel').classList.add('hidden');
            document.getElementById('single-entry-panel').classList.add('hidden');
            if (tabName === 'copypaste') {
                document.getElementById('tab-copypaste-btn').classList.add('active');
                document.getElementById('copypaste-entry-panel').classList.remove('hidden');
            } else if (tabName === 'ai') {
                document.getElementById('tab-ai-btn').classList.add('active');
                document.getElementById('ai-entry-panel').classList.remove('hidden');
            } else {
                document.getElementById('tab-single-btn').classList.add('active');
                document.getElementById('single-entry-panel').classList.remove('hidden');
            }
        }
        async function fetchPendingPrompts() {
            const res = await fetch("/api/pending");
            pendingPrompts = await res.json();
            if (pendingPrompts.length > 0) {
                currentIndex = 0;
                renderActivePrompt();
                document.getElementById("no-pending-message").classList.add("hidden");
                document.getElementById("active-prompt-area").classList.remove("hidden");
            } else {
                document.getElementById("no-pending-message").classList.remove("hidden");
                document.getElementById("active-prompt-area").classList.add("hidden");
            }
        function updateCombinedPromptPreview() {
            if (pendingPrompts.length === 0) return;
            const item = pendingPrompts[currentIndex];
            const additional = document.getElementById("additional-system-instructions").value.trim();
            let systemText = item.system_prompt;
            if (additional) {
                systemText += "\n" + additional;
            }
            const combinedText = `SYSTEM PROMPT:\n${systemText}\n\nUSER PROMPT:\n${item.user_prompt}`;
            document.getElementById("combined-prompt-text").innerText = combinedText;
        }

        function renderActivePrompt() {
            if (pendingPrompts.length === 0) return;
            const item = pendingPrompts[currentIndex];
            document.getElementById("prompt-topic-badge").innerText = item.topic || "General";
            document.getElementById("pending-counter").innerText = `Queue Prompt ${currentIndex + 1} of ${pendingPrompts.length}`;
            updateCombinedPromptPreview();
        }
        function toggleReasoningField() {
            const isChecked = document.getElementById("enable-reasoning").checked;
            if (isChecked) {
                document.getElementById("reasoning-group").classList.remove("hidden");
            } else {
                document.getElementById("reasoning-group").classList.add("hidden");
            }
        }
        async function copyFormattedPrompt() {
            if (pendingPrompts.length === 0) return;
            const item = pendingPrompts[currentIndex];
            const formattedText = `SYSTEM PROMPT:\n${item.system_prompt}\n\nUSER PROMPT:\n${item.user_prompt}`;
            try {
                await navigator.clipboard.writeText(formattedText);
            } catch (e) {}
            await fetch("/api/clipboard/set", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: formattedText })
            });
            showNotification("Prompt block copied to clipboard!");
        }
        async function copyPromptGeneratorInstruction() {
            const topic = document.getElementById("topic").value.trim() || "General";
            const amount = parseInt(document.getElementById("cp-amount").value, 10);
            const instructions = document.getElementById("cp-instructions").value.trim();
            const res = await fetch("/api/copy_prompt_generator_instruction", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ topic, amount, instructions })
            });
            const data = await res.json();
            if (data.status === "success") {
                try {
                    await navigator.clipboard.writeText(data.instruction_text);
                } catch (e) {}
                showNotification("Prompt Creation Request copied to clipboard! Paste into DeepSeek/Web UI.");
            }
        }
        async function parseAndSaveAIPromptOutput() {
            const topic = document.getElementById("topic").value.trim() || "General";
            const rawText = document.getElementById("ai-prompt-output-input").value.trim();
            const modelName = document.getElementById("prompt-model-name").value.trim() || "manual-copy-paste";
            if (!rawText) {
                alert("Please paste the AI output response text.");
                return;
            }
            const res = await fetch("/api/add_prompt", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mode: "batch", topic: topic, raw_text: rawText, model: modelName })
            });
            const data = await res.json();
            if (data.status === "success") {
                document.getElementById("ai-prompt-output-input").value = "";
                showNotification(`Parsed and saved ${data.count} prompts to pending queue!`);
                fetchPendingPrompts();
            } else {
                alert("No valid <prompt><system>...</system><user>...</user></prompt> blocks found.");
            }
        }
        async function pasteClipboard(elementId) {
            try {
                const text = await navigator.clipboard.readText();
                if (text) {
                    document.getElementById(elementId).value = text;
                    return;
                }
            } catch (e) {}
            const res = await fetch("/api/clipboard/get");
            const data = await res.json();
            if (data.text) {
                document.getElementById(elementId).value = data.text;
            }
        }
        async function generatePromptsAI() {
            const topic = document.getElementById("topic").value.trim();
            const amount = parseInt(document.getElementById("amount").value, 10);
            const instructions = document.getElementById("instructions").value.trim();
            const modelName = document.getElementById("prompt-model-name").value.trim() || "manual-copy-paste";
            if (!topic) {
                alert("Please enter a topic.");
                return;
            }
            showNotification("Generating prompts via API... Please wait.");
            const res = await fetch("/api/generate_prompts", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ topic, amount, instructions, model: modelName })
            });
            const data = await res.json();
            if (data.status === "success") {
                showNotification(`Generated ${data.count} prompts and saved to pending queue!`);
                fetchPendingPrompts();
            } else {
                alert(data.message || "Failed to generate prompts.");
            }
        }
        async function addSinglePrompt() {
            const topic = document.getElementById("topic").value.trim() || "General";
            const systemPrompt = document.getElementById("system-prompt-input").value.trim();
            const userPrompt = document.getElementById("user-prompt-input").value.trim();
            const modelName = document.getElementById("prompt-model-name").value.trim() || "manual-copy-paste";
            if (!systemPrompt || !userPrompt) {
                alert("Please provide both system and user prompts.");
                return;
            }
            const res = await fetch("/api/add_prompt", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mode: "single", topic: topic, system_prompt: systemPrompt, user_prompt: userPrompt, model: modelName })
            });
            const data = await res.json();
            if (data.status === "success") {
                document.getElementById("system-prompt-input").value = "";
                document.getElementById("user-prompt-input").value = "";
                showNotification("Prompt added to queue!");
                fetchPendingPrompts();
            }
        }
        async function submitAnswer() {
            if (pendingPrompts.length === 0) return;
            const item = pendingPrompts[currentIndex];
            const modelName = document.getElementById("model-name").value.trim();
            const responseText = document.getElementById("response-input").value.trim();
            const enableReasoning = document.getElementById("enable-reasoning").checked;
            const reasoningText = enableReasoning ? document.getElementById("reasoning-input").value.trim() : "";
            if (!responseText) {
                alert("Please enter the AI response content.");
                return;
            }
            await fetch("/api/submit_answer", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    id: item.id,
                    model: modelName || "manual-web-ui",
                    reasoning: reasoningText,
                    response: responseText
                })
            });
            document.getElementById("reasoning-input").value = "";
            document.getElementById("response-input").value = "";
            showNotification("Answer saved to conversations.json!");
            fetchPendingPrompts();
        }
        async function loadSettingsData() {
            try {
                const resPending = await fetch("/api/pending");
                localPending = await resPending.json();
                const resConvs = await fetch("/api/conversations");
                localConversations = await resConvs.json();
                renderSettingsExplorer();
            } catch (e) {
                console.error(e);
            }
        }
        async function copyMetaPrompt() {
            const topic = document.getElementById("topic").value.trim() || "General";
            const amount = document.getElementById("amount").value || "1";
            const instructions = document.getElementById("instructions").value.trim();
            let priorContext = "";
            try {
                const res = await fetch("/api/prior_prompts?topic=" + encodeURIComponent(topic));
                const data = await res.json();
                if (data.prior_prompts && data.prior_prompts.length > 0) {
                    priorContext = `\n\nPrior prompts (including those currently in the queue/pending) for '${topic}':\n`;
                    data.prior_prompts.forEach((p, i) => {
                        priorContext += `${i+1}. ${p}\n`;
                    });
                    priorContext += `\nCRITICAL: ALWAYS produce COMPLETELY DIFFERENT prompts from prior ones. NEVER reuse core concepts from shown prior examples. Ensure EVERY prompt has a UNIQUE approach.\n`;
                }
            } catch (e) {
                console.error("Failed to fetch prior prompts");
            }
            let promptText = `Generate exactly ${amount} BRAND NEW prompts for: '${topic}'.\n`;
            if (priorContext) {
                promptText += priorContext;
            }
            if (instructions) {
                promptText += `\nAdditional Instructions:\n${instructions}\n`;
            }
            promptText += `
Format requirements:
1. Each prompt must be wrapped in <prompt> tags 
2. Each prompt must contain exactly two tags: 
   - A <system> tag with a system prompt
   - A <user> tag with a user prompt
3. Output ONLY the XML-formatted prompts with no additional text
4. Example format: 
   <prompt>
      <system>System instruction for AI</system>
      <user>User question</user>
   </prompt>
5. For multiple prompts, output them consecutively without separators.`;
            try {
                await navigator.clipboard.writeText(promptText);
            } catch (e) {}
            await fetch("/api/clipboard/set", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: promptText })
            });
            showNotification("Prompt Request copied to clipboard! Paste it into your AI now.");
        }
        function renderSettingsExplorer() {
            const category = document.getElementById("explorer-category").value;
            const format = document.getElementById("explorer-format").value;
            const rawContainer = document.getElementById("explorer-raw-container");
            const humanContainer = document.getElementById("explorer-human-container");
            const dataToRender = category === "pending" ? localPending : localConversations;
            if (format === "raw") {
                rawContainer.classList.remove("hidden");
                humanContainer.classList.add("hidden");
                document.getElementById("explorer-raw-textarea").value = JSON.stringify(dataToRender, null, 2);
            } else {
                rawContainer.classList.add("hidden");
                humanContainer.classList.remove("hidden");
                humanContainer.innerHTML = "";
                if (dataToRender.length === 0) {
                    humanContainer.innerHTML = "<p style='color: var(--text-secondary);'>No records found in this category.</p>";
                    return;
                }
                dataToRender.forEach((item, index) => {
                    const card = document.createElement("div");
                    card.style.backgroundColor = "var(--bg-tertiary)";
                    card.style.border = "1px solid var(--border)";
                    card.style.borderRadius = "6px";
                    card.style.padding = "15px";
                    card.style.marginBottom = "15px";
                    if (category === "pending") {
                        card.innerHTML = `
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <span class="status-badge">${item.topic || "General"}</span>
                                <button onclick="deletePendingPrompt('${item.id}')" style="background-color: #ef4444; font-size: 12px; padding: 6px 12px; border-radius: 4px;">Delete</button>
                            </div>
                            <div style="font-size: 13px; color: var(--text-secondary); margin-bottom: 5px;"><strong>ID:</strong> ${item.id}</div>
                            <div style="margin-bottom: 10px;"><strong>System Prompt:</strong> <pre style="white-space: pre-wrap; font-size: 12px; margin: 5px 0;">${item.system_prompt}</pre></div>
                            <div><strong>User Prompt:</strong> <pre style="white-space: pre-wrap; font-size: 12px; margin: 5px 0;">${item.user_prompt}</pre></div>
                        `;
                    } else {
                        let messagesHtml = "";
                        item.messages.forEach(msg => {
                            let contentToShow = msg.content;
                            if (msg.role === "assistant" && contentToShow.includes("<think>")) {
                                const thinkMatch = contentToShow.match(/<think>([\s\S]*?)<\/think>/);
                                if (thinkMatch) {
                                    const thinkContent = thinkMatch[1].trim();
                                    const responseContent = contentToShow.replace(/<think>[\s\S]*?<\/think>/, "").trim();
                                    contentToShow = `<details style="margin-bottom: 10px; background-color: var(--bg-secondary); padding: 8px; border-radius: 4px;"><summary style="cursor: pointer; font-weight: bold; color: var(--warning);">Reasoning Process</summary><pre style="white-space: pre-wrap; font-size: 12px; margin-top: 5px; color: var(--text-secondary);">${thinkContent}</pre></details>${responseContent}`;
                                }
                            }
                            messagesHtml += `
                                <div style="margin-bottom: 10px;">
                                    <strong style="color: var(--accent); text-transform: uppercase;">${msg.role}:</strong>
                                    <pre style="white-space: pre-wrap; font-size: 12px; margin: 5px 0; background-color: var(--bg-secondary); padding: 8px; border-radius: 4px;">${contentToShow}</pre>
                                </div>
                            `;
                        });
                        card.innerHTML = `
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <span class="status-badge">${item.topic || "General"}</span>
                                <button onclick="deleteCompletedConversation(${index})" style="background-color: #ef4444; font-size: 12px; padding: 6px 12px; border-radius: 4px;">Delete</button>
                            </div>
                            <div style="font-size: 13px; color: var(--text-secondary); margin-bottom: 5px;"><strong>Model:</strong> ${item.model}</div>
                            <div>${messagesHtml}</div>
                        `;
                    }
                    humanContainer.appendChild(card);
                });
            }
        }
        async function copyRawExplorerJson() {
            const jsonText = document.getElementById("explorer-raw-textarea").value;
            try {
                await navigator.clipboard.writeText(jsonText);
                showNotification("JSON copied to clipboard!");
            } catch (e) {
                showNotification("Failed to copy JSON");
            }
        }
        async function deletePendingPrompt(id) {
            if (!confirm("Are you sure you want to delete this pending prompt?")) return;
            try {
                await fetch("/api/delete_pending", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ id })
                });
                showNotification("Pending prompt deleted");
                loadSettingsData();
            } catch (e) {
                console.error(e);
            }
        }
        async function deleteCompletedConversation(index) {
            if (!confirm("Are you sure you want to delete this completed conversation?")) return;
            try {
                await fetch("/api/delete_conversation", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ index })
                });
                showNotification("Completed conversation deleted");
                loadSettingsData();
            } catch (e) {
                console.error(e);
            }
        }
        async function resetQueue(target) {
            const confirmationMsg = target === "pending" 
                ? "Are you sure you want to clear all pending prompts? This cannot be undone." 
                : "Are you sure you want to delete all completed conversations? This cannot be undone.";
            if (!confirm(confirmationMsg)) return;
            try {
                await fetch("/api/reset_file", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ target })
                });
                showNotification("Data reset successfully");
                loadSettingsData();
            } catch (e) {
                console.error(e);
            }
        }
        fetchPendingPrompts();
    </script>
</body>
</html>
"""
@app.route("/")
def index():
    return render_template_string(HTML_CONTENT)
@app.route("/api/pending", methods=["GET"])
def get_pending_prompts():
    pending_items = load_json_data(PENDING_FILE, [])
    filtered_items = [item for item in pending_items if item.get("status") == "pending"]
    return jsonify(filtered_items)
@app.route("/api/conversations", methods=["GET"])
def get_conversations():
    conversations = load_json_data(OUTPUT_FILE, [])
    return jsonify(conversations)
@app.route("/api/clipboard/get", methods=["GET"])
def get_clipboard():
    clipboard_text = get_termux_clipboard()
    return jsonify({"text": clipboard_text})
@app.route("/api/clipboard/set", methods=["POST"])
def set_clipboard():
    request_data = request.get_json() or {}
    text_to_copy = request_data.get("text", "")
    success = set_termux_clipboard(text_to_copy)
    return jsonify({"status": "success" if success else "failed"})
@app.route("/api/prior_prompts", methods=["GET"])
def get_prior_prompts():
    topic = request.args.get("topic", "")
    if not topic:
        return jsonify({"prior_prompts": []})
    prior_prompts = []
    pending_items = load_json_data(PENDING_FILE, [])
    for item in pending_items:
        if item.get("status") == "pending" and item.get("topic", "").lower() == topic.lower():
            user_prompt_content = item.get("user_prompt")
            if user_prompt_content:
                prior_prompts.append(user_prompt_content)
    existing_conversations = load_json_data(OUTPUT_FILE, [])
    for conv in existing_conversations:
        if conv.get("topic", "").lower() == topic.lower():
            for msg in conv.get("messages", []):
                if msg.get("role") == "user":
                    prior_prompts.append(msg.get("content"))
                    break
    return jsonify({"prior_prompts": prior_prompts})
@app.route("/api/copy_prompt_generator_instruction", methods=["POST"])
def copy_prompt_generator_instruction_route():
    request_data = request.get_json() or {}
    topic = request_data.get("topic", "General")
    amount = int(request_data.get("amount", 5))
    instructions = request_data.get("instructions", "")
    instruction_text = construct_prompt_creation_instruction(topic, amount, instructions)
    set_termux_clipboard(instruction_text)
    return jsonify({"status": "success", "instruction_text": instruction_text})
@app.route("/api/generate_prompts", methods=["POST"])
def generate_prompts_route():
    request_data = request.get_json() or {}
    topic = request_data.get("topic", "General")
    amount = int(request_data.get("amount", 1))
    instructions = request_data.get("instructions", "")
    model_name = request_data.get("model")
    try:
        generated_items = execute_ai_prompt_generation(topic, amount, instructions, model_name)
        return jsonify({"status": "success", "count": len(generated_items)})
    except Exception as error_exception:
        return jsonify({"status": "error", "message": str(error_exception)}), 500
@app.route("/api/add_prompt", methods=["POST"])
def add_prompt_route():
    request_data = request.get_json() or {}
    mode = request_data.get("mode", "single")
    topic = request_data.get("topic", "General")
    model_name = request_data.get("model", PROMPTGEN_MODEL)
    if mode == "single":
        system_prompt = request_data.get("system_prompt", "")
        user_prompt = request_data.get("user_prompt", "")
        items = add_single_manual_prompt(topic, system_prompt, user_prompt, model_name)
        return jsonify({"status": "success", "count": len(items)})
    elif mode == "batch":
        raw_text = request_data.get("raw_text", "")
        items = parse_and_add_batch_prompts(topic, raw_text, model_name)
        if items:
            return jsonify({"status": "success", "count": len(items)})
        return jsonify({"status": "error", "message": "No valid XML blocks found"}), 400
    return jsonify({"status": "error", "message": "Invalid mode"}), 400
@app.route("/api/submit_answer", methods=["POST"])
def submit_answer_route():
    request_data = request.get_json() or {}
    prompt_id = request_data.get("id")
    model_name = request_data.get("model", "manual-web-ui")
    reasoning_text = request_data.get("reasoning", "")
    response_text = request_data.get("response", "")
    pending_items = load_json_data(PENDING_FILE, [])
    matched_item = None
    for item in pending_items:
        if str(item.get("id")) == str(prompt_id):
            matched_item = item
            item["status"] = "completed"
            break
    if matched_item:
        save_json_data(PENDING_FILE, pending_items)
        if reasoning_text.strip():
            assistant_content = f"<think>\n{reasoning_text.strip()}\n</think>\n\n{response_text.strip()}"
        else:
            assistant_content = response_text.strip()
        conversation_record = {
            "topic": matched_item.get("topic", "General"),
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": matched_item["system_prompt"],
                    "generation_model": matched_item.get("generation_model", PROMPTGEN_MODEL)
                },
                {
                    "role": "user",
                    "content": matched_item["user_prompt"],
                    "generation_model": matched_item.get("generation_model", PROMPTGEN_MODEL)
                },
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "generation_model": model_name
                }
            ]
        }
        existing_conversations = load_json_data(OUTPUT_FILE, [])
        existing_conversations.append(conversation_record)
        save_json_data(OUTPUT_FILE, existing_conversations)
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Prompt ID not found"}), 404
@app.route("/api/delete_pending", methods=["POST"])
def delete_pending():
    request_data = request.get_json() or {}
    prompt_id = request_data.get("id")
    pending_items = load_json_data(PENDING_FILE, [])
    filtered_items = [item for item in pending_items if str(item.get("id")) != str(prompt_id)]
    save_json_data(PENDING_FILE, filtered_items)
    return jsonify({"status": "success"})
@app.route("/api/delete_conversation", methods=["POST"])
def delete_conversation():
    request_data = request.get_json() or {}
    index = request_data.get("index")
    conversations = load_json_data(OUTPUT_FILE, [])
    if 0 <= index < len(conversations):
        conversations.pop(index)
        save_json_data(OUTPUT_FILE, conversations)
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Index out of range"}), 400
@app.route("/api/reset_file", methods=["POST"])
def reset_file():
    request_data = request.get_json() or {}
    target = request_data.get("target")
    if target == "pending":
        save_json_data(PENDING_FILE, [])
    elif target == "conversations":
        save_json_data(OUTPUT_FILE, [])
    return jsonify({"status": "success"})
if __name__ == "__main__":
    print(f"Flask web server running on http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)