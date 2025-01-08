# 실시간 대화 속 가스라이팅 문장 위험 감지 앱서비스
![image](https://github.com/user-attachments/assets/de34651a-26e3-45f1-82da-cfd8bb9cbfff)
![b](https://github.com/user-attachments/assets/9a25c1fb-0191-4271-baf6-904e7504b64b)
![c](https://github.com/user-attachments/assets/c5394b2d-74d2-4a8d-8111-f76a4787b48f)
![image](https://github.com/eoh9/Gaslighting_chat/assets/62730155/84d4e78c-c42a-46d3-b122-dfa93299ad4e)
![image](https://github.com/eoh9/Gaslighting_chat/assets/62730155/5af50c0e-a9e5-426a-a2d8-ca52e9fc6e89)
![image](https://github.com/eoh9/Gaslighting_chat/assets/62730155/6141a18e-f3c3-4303-9011-9a20a48e51e7)

현재 시중의 챗봇 및 클린봇들은 욕설이나 혐오표현 등의 부정적인 언어를 탐지하는 데 집중하고 있지만, 가스라이팅과 같은 더 섬세하고 복잡한 형태의 부적절한 대화를 식별하는 데에는 한계가 있습니다. 가스라이팅은 직장, 학교, 가정, 연인 관계 등 다양한 수직 관계에서 발생할 수 있으며, 이를 탐지하고 경고하는 기능은 아직 대중적인 서비스에서 제공되지 않고 있습니다. 이 프로젝트는 사용자가 대화하는 도중 가스라이팅이 의심되는 문장을 식별하고, 해당 문장을 하이라이트하여 사용자가 인지할 수 있도록 돕는 기능을 개발하는 것을 목표로 합니다. 이를 통해 가스라이팅에 대한 경각심을 높이고, 사전에 예방할 수 있는 방법을 제공하고자 합니다. 현재 국내에서는 가스라이팅에 대한 법적 처벌 규정이 명확히 정의되어 있지 않지만, 이 주제는 사회적으로 중요하게 다뤄지고 있으며, 우리는 이에 대한 인식을 높이고 예방하는 데 기여하고자 합니다.

## 데이터 수집
1. 공공데이터 & AI Hub
2. 비정형 데이터: 디시인사이트, 네이버카페, 다음카페, 네이트판, 인벤, 펨코, 인스타그램, 페이스북 등.
3. 구글 대화 이미지 크로링 및 텍스트 추출
4. 데이터셋 자체 제작(전문가의 자문을 받아 검증을 완료, LGG그래프 활용)

## 데이터 증강
1. KoGPT fine-tuning

## 가스라이팅 문장 분류
1. KoBERT
