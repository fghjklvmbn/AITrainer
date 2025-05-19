# AITrainer
인공지능 학습 모듈(연습용)

![image](https://github.com/user-attachments/assets/2501d559-aada-431b-995e-48c11372ce9d)

HuggingFace 기반의 학습도구(transformer, peft, datasets)로 제작되었으며, 모델은 Qwen 2.5 버전 기반으로 FineTuning 하는 코드입니다.
추후 학습에 필요한 변수는 변경 가능하도록 UI 혹은 yaml 파일로 바꿀 예정이며, 차후 데이터 모델이 충족된다면 Hugging Face으로 올릴 예정입니다.

**학습모델**
 - MLP-KTLim/llama-3-Korean-Bllossom-8B
 - beomi/KoAlpaca-llama-1-7b
 - Qwen/Qwen2.5-0.5B
 - Qwen/Qwen2.5-7B

**준비사항**
 - 아나콘다(권장)
 - python 3.12+

**가이드**
1. 다운로드 후 압축을 푼다.
2. cmd, ssh등을 통해 압축을 푼 폴더로 들어간다.
3. ``` pip install -r requirements.txt ``` 명령어를 이용해서 설치해 준다.
4. python *.py를 통해 파이썬 파일을 실행한다.

**config 적용방법**
 - config 파일은 오로지 train.py 파일으로만 사용할 수 있습니다.
 - train.py 파일은 config파일이 없으면 실행이 불가하니 이점 유의해주시기 바랍니다.
 - 명령어 : 
```python train.py -c ./config/파일속_설정파일.yaml```
 - 이러한 명령어로 train.py를 실행시킬수 있습니다.

파일설명
 - **requirements.txt** - 학습에 필요한 모듈과 API 구동 모듈을 모두 모아 놓은 파일입니다, 
가이드에 말한 것 처럼 ``` pip install -r requirements.txt ``` 명령어로 필요한 모듈을 모두 설치해서 쓸 수 있습니다.

 - **train.py** - 학습용 파이썬 파일입니다.

 - **model_run.py** - 학습 완료시 모델을 구동 가능하게 하는 파이썬 모델입니다.

 - **train_beta.py** - (테스트중)차세대 코드가 포함된 학습파일입니다.

data폴더

 - **data_all.py** - 학습에 필요한 목업 데이터 파일입니다.

 - **load_json.py** - (테스트중)json 파일을 파이썬이 지원하는 데이터셋으로 바꿔주는 함수가 집약된 파이썬 파일입니다.

json_files 폴더

 - **character.json** - 각 인물의 성격을 구체적으로 구현하는데 도움이 되도록 등장인물의 상세정보를 모두 집약시킨 데이터셋 파일입니다.
   
 - **story.json** - 각 제목과 줄거리, 태그를 원활하게 생성하도록 도움을 주는 데이터셋 파일입니다.

 - **write_style.json** - 동화가 생성될때 문체를 학습시키도록 유도하고 도움을 주는 데이터셋 파일입니다.

server_model 폴더

 - **app.py** - flask를 이용한 API 가동 파일입니다.

 - **utils.py** - API 단에서 들어온 프롬프트를 모델으로 구동하고, 그 결과를 전달하는 파일입니다.

config 폴더

 - **config_0.5b.yaml** : qwen 2.5 버전 파라미터 5억개 버전의 설정파일입니다.

 - **config_7b.yaml** : qwen 2.5 버전 파라미터 70억개 버전의 설정파일입니다.

 - **config_14b.yaml** : qwen 2.5 버전 파라미터 140억개 버전의 설정파일입니다.
