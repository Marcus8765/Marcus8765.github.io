// index.js
// A minimal AI agent: a program with a loop.
//
//   read input -> decide (ask the model) -> act (run tools if asked)
//   -> observe (feed results back) -> repeat
//
// This follows the pattern from "Building an AI Agent from Scratch"
// (Juntao Qiu, The Pragmatic Developer): start with a bare REPL, wire it
// up to a local model via Ollama, then give the model tools it can call.

import "dotenv/config";
import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { ollamaChat } from "./ollama.js";
import { toolDefinitions, runToolCall } from "./tools/index.js";

const EXIT_WORDS = new Set(["exit", "quit", ":q", "bye"]);

function shouldExit(line) {
  return EXIT_WORDS.has(line.trim().toLowerCase());
}

async function main() {
  const rl = createInterface({ input, output });

  /** @type {{ role: string, content: string, tool_calls?: any[], tool_call_id?: string, name?: string }[]} */
  const messages = [
    {
      role: "system",
      content:
        "You are a helpful assistant with access to tools. Use the get_weather tool when asked about weather; otherwise answer normally.",
    },
  ];

  output.write(
    'Agent ready. Type a message ("exit" to quit). Try: "What\'s the weather in Tokyo?"\n\n'
  );

  while (true) {
    let line;
    try {
      line = await rl.question("You> ");
    } catch {
      // Input stream closed (e.g. piped stdin hit EOF, or Ctrl+D) — exit quietly.
      break;
    }
    const trimmed = line.trim();

    if (!trimmed) continue;
    if (shouldExit(trimmed)) break;

    messages.push({ role: "user", content: trimmed });

    try {
      await decideActObserve(messages);
    } catch (err) {
      output.write(`\n[error] ${err.message}\n\n`);
    }
  }

  rl.close();
}

/**
 * One full turn of the loop: ask the model, run any requested tools,
 * feed results back, and keep going until the model gives a plain answer.
 */
async function decideActObserve(messages) {
  // Cap tool-calling rounds so a confused model can't loop forever.
  const MAX_ROUNDS = 5;

  for (let round = 0; round < MAX_ROUNDS; round++) {
    // 1) Decide: send the conversation + tool definitions to the model.
    const data = await ollamaChat(messages, toolDefinitions);
    const assistant = data?.message;

    if (!assistant) {
      output.write("\nAssistant> (no response from model — check Ollama logs)\n\n");
      return;
    }

    const toolCalls = assistant.tool_calls;

    if (!toolCalls || toolCalls.length === 0) {
      // No tool requested: this is the final answer for this turn.
      const reply = assistant.content?.trim?.() || "(empty response)";
      messages.push({ role: "assistant", content: reply });
      output.write(`\nAssistant> ${reply}\n\n`);
      return;
    }

    // 2) Act: the model wants to call one or more tools. Run each one.
    messages.push({
      role: "assistant",
      content: assistant.content ?? "",
      tool_calls: toolCalls,
    });

    for (const toolCall of toolCalls) {
      const toolName = toolCall.function?.name;
      output.write(`\n[tool] calling ${toolName}(${JSON.stringify(toolCall.function?.arguments)})\n`);

      const resultJson = await runToolCall(toolCall);
      output.write(`[tool] ${toolName} -> ${resultJson}\n`);

      // 3) Observe: feed the tool's result back into the conversation.
      messages.push({
        role: "tool",
        name: toolName,
        content: resultJson,
      });
    }

    // 4) Repeat: loop back and let the model use the tool result(s)
    // to produce a real answer (or call another tool).
  }

  output.write("\nAssistant> (gave up after too many tool-call rounds)\n\n");
}

main();
