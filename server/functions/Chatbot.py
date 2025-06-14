import torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM

class Chatbot:
    def __init__(self, model_path="Qwen/Qwen3-1.7B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.history = []  # 대화 기록 저장

    def strip_think_tags(text: str) -> str:
        return re.sub(r"user.*?</think>\n*", "", text, flags=re.DOTALL)
    
    def format_prompt(self):
        """이전 대화 기록을 기반으로 프롬프트 생성"""
        return self.tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

    def chat(self, user_input):
        """사용자 입력을 받아 대화 기록을 업데이트하고, 모델로 응답 생성"""
        # 사용자 메시지 추가
        self.history.append({"role": "user", "content": user_input})
        
        # 프롬프트 생성
        text = self.format_prompt()
        
        # 토큰화
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # 텍스트 생성
        generated_ids = self.model.generate(
            **model_inputs,
            temperature=0.5,
            max_new_tokens=32768
        )
        
        # 생성된 응답 추출
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # 응답 추가
        self.history.append({"role": "assistant", "content": response})

        response = self.strip_think_tags(response)  # think 태그 제거
        return response