from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# 모델 및 토크나이저 로딩
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# 언어 코드 매핑
LANG_CODE_MAP = {
    "ko": "ko_KR",
    "en": "en_XX",
    "ja": "ja_XX",
    "zh": "zh_CN",
    "fr": "fr_XX",
    "de": "de_DE"
}

def translate_text(text, target_language):
    """
    MBart50을 사용한 번역 함수
    :param text: 입력 텍스트 (한국어 기준)
    :param target_language: 목적 언어 (ISO 코드, 예: 'en', 'fr')
    :return: 번역된 문자열
    """
    src_lang = "ko"  # 현재 기준은 한국어 입력 고정
    tgt_lang = target_language.lower()

    if tgt_lang not in LANG_CODE_MAP:
        return f"지원하지 않는 언어 코드입니다: {target_language}"

    tokenizer.src_lang = LANG_CODE_MAP[src_lang]

    # 입력 텍스트 토크나이징
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # 번역 실행
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[LANG_CODE_MAP[tgt_lang]],
        max_length=512,
        num_beams=5,
        no_repeat_ngram_size=2
    )

    # 디코딩 후 출력
    translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated