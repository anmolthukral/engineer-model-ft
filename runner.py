from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "./merged_model"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

def ask(question: str) -> str:
    # Model was fine-tuned on raw "### User:/### Assistant:" text, not ChatML —
    # apply_chat_template() would feed it a prompt format it never saw in training.
    prompt = f"### User:\n{question}\n### Assistant:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    question = "Act as a teacher and teach me step by step how to flatten an array in JavaScript"
    print(ask(question))
