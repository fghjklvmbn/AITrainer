# 1. 데이터 준비: 동화책 문체 예시 데이터
data = [
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "한참을 걸어가던 아기 곰이 숲 속에서 작은 새를 만났어요."},
    {"instruction": "주인공이 용감하게 행동하는 장면을 써줘.", "input": "", "output": "용감한 소녀는 무서움을 떨치고 어두운 동굴 속으로 들어갔어요."},
    {"instruction": "마법사가 등장하는 이야기를 시작해줘.", "input": "", "output": "오래된 성에 살던 마법사는 어느 날 신비한 책을 발견했어요."},
    {"instruction": "행복한 결말을 가진 이야기를 써줘.", "input": "", "output": "모든 친구들이 함께 춤을 추며 행복한 시간을 보냈답니다."}, 
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 다람쥐는 겨울을 준비하며 도토리를 모았어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "별 하나가 빛을 잃자 다른 별들이 걱정했어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "강물은 바다를 향해 천천히 흘러갔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "바람은 나뭇잎을 흔들며 인사를 했어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "마법의 열쇠는 오래된 문을 열었어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "아이들은 들판에서 꽃을 모으며 웃었어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "눈이 내린 숲 속에서는 모든 것이 조용했어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 배는 파도를 타며 멀리 떠나갔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "달빛이 호수에 비치자 물고기들이 반짝였어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "소년은 하늘을 보며 별을 세기 시작했어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 불꽃이 어두운 방을 밝혔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "나무는 자신의 가지에 달린 열매를 자랑했어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "요정은 꽃잎을 타고 날아다녔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "별빛 아래에서 두 친구는 비밀을 나누었어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "강아지는 주인을 만나기 위해 멀리 달려갔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "달은 고독한 사람에게 따뜻한 빛을 보내주었어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 새는 둥지를 짓기 위해 나뭇가지를 모았어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "마법사는 자신의 지팡이로 기적을 만들어냈어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "바다는 파도를 일으키며 노래를 불렀어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 불꽃은 어둠 속에서도 빛났어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "눈 덮인 산길을 따라 소년이 걸어갔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "별은 밤하늘에서 반짝이며 이야기를 들려주었어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 배는 바람을 타고 멀리 떠나갔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "나무는 뿌리를 깊게 내리며 살아남았어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "바람은 메아리처럼 이야기를 전했어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "별은 자신의 빛으로 세상을 따뜻하게 했어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "강물은 바다를 향해 천천히 흘러갔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 새는 둥지를 짓기 위해 나뭇가지를 모았어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "마법사는 자신의 지팡이로 기적을 만들어냈어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "바다는 파도를 일으키며 노래를 불렀어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 불꽃은 어둠 속에서도 빛났어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "눈 덮인 산길을 따라 소년이 걸어갔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "별은 밤하늘에서 반짝이며 이야기를 들려주었어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 배는 바람을 타고 멀리 떠나갔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "나무는 뿌리를 깊게 내리며 살아남았어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "바람은 메아리처럼 이야기를 전했어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "별은 자신의 빛으로 세상을 따뜻하게 했어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "강물은 바다를 향해 천천히 흘러갔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 새는 둥지를 짓기 위해 나뭇가지를 모았어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "마법사는 자신의 지팡이로 기적을 만들어냈어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "바다는 파도를 일으키며 노래를 불렀어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 불꽃은 어둠 속에서도 빛났어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "눈 덮인 산길을 따라 소년이 걸어갔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "별은 밤하늘에서 반짝이며 이야기를 들려주었어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 배는 바람을 타고 멀리 떠나갔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "나무는 뿌리를 깊게 내리며 살아남았어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "바람은 메아리처럼 이야기를 전했어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "별은 자신의 빛으로 세상을 따뜻하게 했어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "강물은 바다를 향해 천천히 흘러갔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 새는 둥지를 짓기 위해 나뭇가지를 모았어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "마법사는 자신의 지팡이로 기적을 만들어냈어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "바다는 파도를 일으키며 노래를 불렀어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 불꽃은 어둠 속에서도 빛났어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "소녀는 꽃밭에서 꿈을 꾸며 춤을 추었어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "바람은 메아리처럼 이야기를 전했어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "별은 자신의 빛으로 세상을 따뜻하게 했어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "강물은 바다를 향해 천천히 흘러갔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 새는 둥지를 짓기 위해 나뭇가지를 모았어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "마법사는 자신의 지팡이로 기적을 만들어냈어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "바다는 파도를 일으키며 노래를 불렀어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 불꽃은 어둠 속에서도 빛났어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "눈 덮인 산길을 따라 소년이 걸어갔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "별은 밤하늘에서 반짝이며 이야기를 들려주었어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 배는 바람을 타고 멀리 떠나갔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "나무는 뿌리를 깊게 내리며 살아남았어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "바람은 나뭇잎을 흔들며 인사를 했어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "마법의 책은 비밀스러운 이야기를 들려주었어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 요정은 꽃잎을 타고 날아다녔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "별 하나가 빛을 잃자 다른 별들이 걱정했어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "강아지는 주인을 만나기 위해 멀리 달려갔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "달은 고독한 사람에게 따뜻한 빛을 보내주었어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 불꽃이 어두운 방을 밝혔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "바다는 파도를 일으키며 이야기를 들려주었어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "나무는 자신의 가지에 달린 열매를 자랑했어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 요정은 꽃잎을 타고 날아다녔어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "별빛 아래에서 두 친구는 비밀을 나누었어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "바람은 나뭇잎을 흔들며 인사를 했어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "마법의 열쇠는 오래된 문을 열었어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "아이는 하늘을 보며 별을 세기 시작했어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "작은 배는 바다를 건너 새로운 섬을 발견했어요."},
    {"instruction": "동화책 문체로 문장을 생성해줘", "input": "", "output": "나무는 뿌리를 깊숙이 내리며 살아남았어요."}
]