# Archivit
#### Project of 2024 google solution challenge

![alt text](https://github.com/AGILE-CK/.github/raw/main/images/KakaoTalk_Photo_2024-02-23-22-07-22%20002.jpeg)

## MEMBERS
| Ko EunJin | Lee Sunggu  | Lim ChaeHyeong | Lao ChingSan |
|---|---|---|---|
| Lead / Design | Frontend / Backend | Frontend | AI / Frontend |
| KOR | KOR | KOR | CAM |
---
## Targeting of UN - SDGs

<!-- ![alt text](image-1.png) ![alt text](image-3.png) ![alt text](image-4.png) -->
![alt text](https://github.com/AGILE-CK/.github/raw/main/images/KakaoTalk_Photo_2024-02-23-22-07-22%20003.jpeg)
#### About our solution 
In our current korea society, various incidents of violence have emerged as a serious social issue. The Archivit app aims to help vulnerable individuals live a healthy life without worrying about tomorrow amid these incidents.

We provide users with an AI-powered recording feature to adopt evidence of violence when they are exposed to such situations. Even in situations where victims cannot manually record, AI recognizes the violent situation and automatically proceeds with recording, which can be immensely helpful for victims in legal proceedings.

We offer information on how victims of verbal and physical violence can receive assistance and overcome their situations. Through various platforms where victims can seek redress, we make it easy to find information that may have been difficult to obtain through internet searches.

Furthermore, we provide a template feature that allows victims to record their daily experiences. Through this feature, victims can document events in writing, enabling them to adopt more accurate evidence and alleviate stress by expressing concerns they may not have been able to share with anyone before.

Finally, when used as evidence, we have added a feature that utilizes AI to convert voice recordings into text, making past situations easier to report and understand. Through these measures, we believe that victims of violence can resolve problematic situations without injustice.

___

## App Overview
<!-- <img src="image-5.png" alt="alt text" width="390" height="844">

<img src="image-6.png" alt="alt text" width="390" height="844">

<img src="image-7.png" alt="alt text" width="390" height="844">

<img src="image-8.png" alt="alt text" width="390" height="844">

<img src="image-9.png" alt="alt text" width="390" height="844"> -->

![alt text](https://github.com/AGILE-CK/.github/raw/main/images/KakaoTalk_Photo_2024-02-23-22-07-23%20005.jpeg)


![alt text](https://github.com/AGILE-CK/.github/raw/main/images/KakaoTalk_Photo_2024-02-23-22-07-23%20006.jpeg)



---


## How to Start

#### Frontend
You should install Flutter.
(My Version: Flutter SDK: 3.19.0 , Dart: 3.3.0)

<b>
Our Flutter project is only available for use in the iOS environment.
</b>
So before, command 'flutter run'. Please, Launch the IOS Emulator or connect with IOS phone.

```
flutter run // and choose IOS Platform.
```
#### Backend
You should install Go.(My Version: go1.21.5)

Go to the main.go directory.


```
docker compose up -d
go run main.go
```

#### AI
To run this project, you need Python 3.7 or newer. It's recommended to use a virtual environment.

1. Install google-generativeai package:

```
pip install -q -U google-generativeai
```

2. Install dependencies:
```
pip install -r requirements.txt
```
Requirements
- FastAPI
- PyTorch
- Transformers
- Pydub
- Soynlp
- Pyannote.audio
- Google Cloud Generative AI services (for summarization)
- Librosa

Ensure you have the necessary API keys and tokens for services such as Hugging Face and Google Cloud.

3. Start the server with:
```
uvicorn main:app --reload
```
Endpoints
POST /violent-speech-detection/: Detects violent content in the speech.
POST /calm-situation-detection/: Identifies calm situations from audio.
POST /transcribe/: Performs speaker diarization and transcription.
POST /summarize/: Summarizes the provided text.

---



## About Implement
#### Tech Stack
###### - Frontend 
* Flutter 
* Getx
* flutter_sound
* flutter_background_service 
* etc

###### - Backend
* Golang
* Gin-Gonic
* Gorm
* Swagger
* Mysql
* GCP App Engine


###### - AI
* FastAPI: For building efficient, asynchronous REST APIs that serve our machine learning models.
* PyTorch & Transformers: For loading and serving state-of-the-art NLP and audio processing models.
* Librosa & Pydub: For audio file manipulation and format conversion.
* TensorFlow & Hugging Face Pipelines: For emotion detection and speaker diarization.

#### Project Architecture
![alt text](https://github.com/AGILE-CK/.github/raw/main/images/KakaoTalk_Photo_2024-02-23-22-07-23%20004.jpeg)


#### Server URL 

* [Backend API Docs](https://agile-dev-dot-primeval-span-410215.du.r.appspot.com/swagger/index.html#/)

* [AI API Docs](https://final-apcfknrtba-du.a.run.app/docs)

----
## Improvement things

#### Planning
The methods for victims to receive remedies are very diverse and hard to find. Even if you succeed in finding it, the procedure is complex. We wanted to solve this by guiding the process through a chatbot and allowing us to create a remedy application that fits the templates of each institution through the evidence we have collected and AI. However, during this Solution Challenge period, we did not have much time and could not fully implement it.


* [Source](https://docs.google.com/spreadsheets/d/1hcungJAGTACJApmnxmV_1gB_2RpvLb0A/edit?usp=sharing&ouid=103564838690673927306&rtpof=true&sd=true)
#### Frontend 

* Currently, this project only supports iOS. In the case of Android, there is a lack of functionality for background services. It would be great if this could be further supplemented.


#### Backend
* Firebase Replacement: I'm currently using the Go language. However, most of the current backend APIs can be replaced with Firebase. If I had more time, I would like to replace it.
* GATEWAY API: Currently, JWT security has not been properly applied to AI. Through the GATEWAY API, token verification can be performed. I want to implement this feature.
* GRPC: Currently, in violent situations, we are recording every 10 seconds and getting verification from AI. If we use GRPC, more real-time verification will be possible.

#### AI
* Enhance Model Accuracy: Continually retrain models with diverse, updated datasets to improve recognition accuracy and reduce bias.

* Increase Processing Speed: Use model optimization techniques like quantization and parallel processing to ensure real-time performance even under high load.

* Strengthen Privacy and Security: Implement voice anonymization and secure data handling to protect user privacy and data integrity.

* Expand Features and Languages: Develop context-aware models and broaden language support to cater to a global user base.


----
## FeedBack
#### Frontend 
 Flutter did not support multiple recordings. If I had realized this fact sooner and thought of an alternative(Like GRPC), I could have achieved much better results. And I felt that communication within the team is important.

#### Backend
During the last GDSC Solution Challenge project, I tried using a framework called Gin-Gonic in Golang out of personal interest, without considering the plan. However, in terms of planning, our project's backend was very simple, and most of it could have been resolved at the Firebase level. I realized that it is also important to consider how to implement considering the planning. I also regret not being able to finish applying both the GATEWAY API and GRPC.

#### AI
During the development of our project for the 2024 Google Solution Challenge, choosing the right AI model and integrating it with our app were our main challenges. Balancing accuracy with efficiency and scalability required careful consideration, as we aimed for a solution that offered real-world reliability without compromising performance. The integration process demanded a meticulous approach to ensure seamless communication between our AI API and the app, prioritizing user experience. This journey taught us the importance of adaptability and the value of community feedback in navigating technical challenges, reinforcing our commitment to continuous improvement and innovation in our quest to make a meaningful impact.

#Thank you
