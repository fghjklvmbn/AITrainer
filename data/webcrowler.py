import requests
from bs4 import BeautifulSoup

urls = [
    'https://redbadastory.tistory.com/67',
    'https://redbadastory.tistory.com/68'
]

i = 1
output_file = "동화.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for url in urls:
        res = requests.get(url)
        print(f"Processing {url} - Status: {res.status_code}")

        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            content_area = soup.select_one("#article-view > div.tt_article_useless_p_margin.contents_style")

            if content_area:
                tags = content_area.find_all("p")
                texts = [tag.get_text().strip() for tag in tags if tag.get_text(strip=True)]
                unique_texts = list(dict.fromkeys(texts))  # 중복 제거

                if unique_texts:
                    f.write(f"---------#{i}--------\n")
                    for text in unique_texts:
                        f.write(text + "\n")
                    f.write("\n\n\n\n")
                    i += 1
                else:
                    print("콘텐츠가 비어 있습니다.")
            else:
                print("콘텐츠 영역을 찾지 못했습니다.")
        else:
            print(f"상태 코드: {res.status_code}")

print("텍스트를 동화.txt에 저장했습니다.")
