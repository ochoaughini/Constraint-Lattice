# Using Constraint Lattice With the 🤗 Transformers Library

This short guide shows how to combine Constraint Lattice with any model that is available via the 🤗 *transformers* ecosystem.  We illustrate the workflow with Google’s **Gemma-2 (2-B Instruction-Tuned)** model, but you can swap in any other causal-LM.

---

## Authenticate With Hugging Face Hub (Optional)
If you are downloading a gated or license-restricted checkpoint you may need to login first:

```bash
huggingface-cli login  # follow the browser prompt
```

## Quick One-Liner With `pipeline`
For simple experiments you can rely on the high-level `pipeline` API:

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="google/gemma-2-2b-it")

messages = [
    {"role": "user", "content": "Who are you?"},
]

print(pipe(messages))
```

> **Tip** — wrap the generated string with `apply_constraints()` (or the SDK / REST endpoint) to enforce all your safety rules.

## Load Model and Tokenizer Explicitly
For fine-grained control you may want to instantiate the model and tokenizer yourself:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-2-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = AutoModelForCausalLM.from_pretrained(model_id)
```

Now you can craft prompts, call `model.generate()`, decode, and pass the raw output through Constraint Lattice just like any other text source.

---

### Quick Links
* [Gemma model card](https://huggingface.co/google/gemma-2-2b-it)
* [Pipeline documentation](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines)
* [transformers tutorials](https://huggingface.co/learn/nlp-course/chapter1/1)

---

*Last updated: 2025-06-29*
