seeimport litserve as ls
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Define constants
SYSTEM_PROMPT = "<|system|>\nYou are a senior Python developer. Provide clear, correct, well-commented code.<|end|>\n\n"
USER_TOKEN = "<|user|>\n"
ASSISTANT_TOKEN = "<|assistant|>\n"
END_TOKEN = "<|end|>"

class LLMAPI(ls.LitAPI):
    def setup(self, device):
        # Load model and tokenizer
        model_name = "mohsin416/phi3-python-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Move model to device (GPU/CPU)
        if device != "cpu":
            self.model.to(device)
        self.model.eval()

    def _format_prompt(self, prompt: str) -> str:
        return f"{SYSTEM_PROMPT}{USER_TOKEN}{prompt}{END_TOKEN}{ASSISTANT_TOKEN}"

    # 1. DECODE: Extract prompt from request and prepare tensors
    def decode_request(self, request):
        prompt = request.get("prompt", "")
        formatted_prompt = self._format_prompt(prompt)
        
        # Create input tensors here
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        return inputs.input_ids.to(self.model.device)

    # 2. PREDICT: Pure model inference
    def predict(self, input_ids):
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.2,
                top_p=0.80,
                do_sample=True,
                eos_token_id=self.tokenizer.convert_tokens_to_ids(END_TOKEN),
                pad_token_id=self.tokenizer.eos_token_id
            )
        return outputs[0]

    def encode_response(self, output):
        generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
        
        if ASSISTANT_TOKEN in generated_text:
            generated_code = generated_text.split(ASSISTANT_TOKEN)[1].strip()
            # Clean up trailing end tokens if they exist in the split
            if END_TOKEN in generated_code:
                generated_code = generated_code.split(END_TOKEN)[0].strip()
        else:
            generated_code = generated_text
            
        return {"response": generated_code}

if __name__ == "__main__":q
    server = ls.LitServer(LLMAPI(), accelerator="auto")
    server.run(port=8000)
