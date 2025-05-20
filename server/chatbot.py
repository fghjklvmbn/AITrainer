from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Chatbot:
    def __init__(self, model_name="Qwen/Qwen2.5-7B"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        self.history = []  # 대화 기록: [{"role": "user", "content": "..."}, ...]

    def format_history(self):
        """대화 기록을 모델 입력 형식으로 변환"""
        prompt = ""
        for msg in self.history:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            else:
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant: "
        return prompt

    # def generate_response(self, user_input):
    #     """사용자 입력을 기반으로 응답 생성"""
    #     self.history.append({"role": "user", "content": user_input})
    #     prompt = self.format_history()
        
    #     inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    #     outputs = self.model.generate(
    #         inputs["input_ids"],
    #         max_length=128,
    #         num_return_sequences=1,
    #         temperature=0.7,
    #         top_p=0.95,
    #         do_sample=True
    #     )
    #     response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    #     # 응답을 대화 기록에 추가
    #     assistant_response = response[len(prompt):].strip()
    #     self.history.append({"role": "assistant", "content": assistant_response})
    #     return assistant_response

    def generate_response(self, user_input):
        self.history.append({"role": "user", "content": user_input})
        prompt = self.format_history()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=inputs["input_ids"].shape[1] + 200,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id  # pad_token 지정
        )

        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        # 프롬프트 이후의 응답만 추출
        if full_output.startswith(prompt):
            assistant_response = full_output[len(prompt):].strip()
        else:
            # fallback: 전체에서 마지막 Assistant 이후 부분 추출
            assistant_response = full_output.split("Assistant:")[-1].strip()

        self.history.append({"role": "assistant", "content": assistant_response})
        return assistant_response

    def reset(self):
        """대화 기록 초기화"""
        self.history = []