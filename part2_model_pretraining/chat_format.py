"""
The chat template: how an instruction and its answer are laid out as text.

CHECKED against Santosh's sft_template.txt (27 Oct): identical.
    <|im_start|>user\n{instruction}\n\n{context}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
    - "\n\n{context}" is dropped when context is empty; all fields are .strip()-ed
    - prompt and response are tokenized SEPARATELY, then joined
    - stop token for generation: <|im_end|> (id 3) only
Don't change these strings: SFT training and Sree's predictions both use them.

Special tokens (from Santosh): <|endoftext|>=0, <|pad|>=1, <|im_start|>=2, <|im_end|>=3
"""

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
IM_END_ID = 3
PAD_ID = 1

# The part the model READS (no loss here). Ends right where the answer starts.
PROMPT_TEMPLATE = IM_START + "user\n{user_text}" + IM_END + "\n" + IM_START + "assistant\n"
# The part the model must LEARN TO WRITE (loss here). Ends with the stop token.
RESPONSE_TEMPLATE = "{response}" + IM_END


def user_text_from_dolly(row):
    """Dolly rows have 'instruction' and an optional 'context' (a passage to read)."""
    instruction = row["instruction"].strip()
    context = (row.get("context") or "").strip()
    if context:
        return instruction + "\n\n" + context
    return instruction


def format_prompt(row):
    """
    Turn a data row into the prompt string. Accepts:
      - a row that already has a formatted 'prompt' string (used as-is), or
      - a Dolly-style row with 'instruction' (+ optional 'context').
    """
    if "prompt" in row and isinstance(row["prompt"], str):
        return row["prompt"]
    return PROMPT_TEMPLATE.format(user_text=user_text_from_dolly(row))


def format_response(row):
    return RESPONSE_TEMPLATE.format(response=row["response"].strip())
