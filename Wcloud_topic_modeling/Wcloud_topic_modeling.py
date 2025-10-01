# ===============================
# 0. 라이브러리 설치 (처음 한 번만)
# ===============================
# !pip install pandas openpyxl konlpy wordcloud matplotlib gensim

import pandas as pd
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim import corpora, models

# ===============================
# 1. 엑셀 불러오기
# ===============================
# 엑셀 파일: 'articles.xlsx'
# 텍스트가 들어있는 컬럼 이름은 'content'라고 가정
df = pd.read_excel("/home/alpaco/kimcy/Office_AI_Agent_System/Wcloud_topic_modeling/Wcloud_topic_modeling.csv")
texts = df["content"].dropna().tolist()

# ===============================
# 2. 형태소 분석 (명사 추출)
# ===============================
okt = Okt()
tokenized_texts = []

for text in texts:
    tokens = okt.nouns(text)  # 명사만 추출
    tokens = [t for t in tokens if len(t) > 1]  # 한 글자 단어 제거
    tokenized_texts.append(tokens)

# ===============================
# 3. 워드클라우드
# ===============================
all_tokens = [token for sublist in tokenized_texts for token in sublist]
counter = Counter(all_tokens)

wc = WordCloud(
    font_path="/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # 한글 폰트 경로
    background_color="white",
    width=800,
    height=600
)
plt.figure(figsize=(10,8))
plt.imshow(wc.generate_from_frequencies(counter), interpolation="bilinear")
plt.axis("off")
plt.show()

# ===============================
# 4. 토픽 모델링 (LDA)
# ===============================
# 단어 사전 & 말뭉치 생성
dictionary = corpora.Dictionary(tokenized_texts)
corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

# LDA 모델 학습 (토픽 5개 가정)
lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

# 토픽 출력
for idx, topic in lda_model.print_topics(num_words=10):
    print(f"토픽 {idx}: {topic}")
