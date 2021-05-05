# 조직 병리 이미지의 암 부위 시각화 및 웹 서비스 제작

```
💡 팀 구성 : Web Front-End & Back-End, AI Developing
```



## Capstone Project 개요

```
💡 - 큰 사이즈의 병리 이미지에서의 Annotation은 상당한 시간과 노동력이 소모된다.
💡 - 현재 프로그램으로 제공되는 Whole Slide Image는 뷰어나 Annotation을 병원 내 서버 저장소나 로컬을 통해 진행  			돼, 저장공간 등을 소모하게 된다. 
```



병리 이미지를 다루면서 실제 병리과 전문의 분들께 이야기를 들어보거나 병리이미지와 관련된 논문을 읽게 되면 자주 나오는 이야기는 **Annotation**입니다. 큰 사이즈의 병리 이미지를 Zoom-in, Zoom-out을 하며 일일이 확인해야하며 이에 따른 Cancer Segmentation, Bounding Box등 다양한 Annotation을 진행하게 되는데, 상당한 **시간과 노동력**이 소모됩니다.

→ Annotation이 진행되는 Tool에서 성능이 좋은 AI model이 있을 경우, 참고용으로 확인하며 Annotation을 진행하여 **시간 절약**이 가능합니다.

현재 병리 Annotation Tool은 로컬 환경으로 이미지를 옮겨서 Annotation을 해야합니다. 이는 저장공간 등의 로컬환경에 영향을 끼치는 문제가 존재합니다.

→ Annotation이 진행되는 환경이 Web Server단에서 제공이 될 경우, **메모리, 저장공간의 제약**을 상대적으로 덜 받을 수 있고, 인터넷만 있으면 **작업환경을 구축**할 수 있습니다.



### 🚀 현재까지의 시행, 제품

**QuPath | Quantitative Pathology & Bioimage Analysis**

------

QuPath는 병리 Annotation tool을 제공하고 있습니다. 로컬의 이미지등을 프로그램에 업로드하여 Annotation을 진행하고 Annotation file을 저장하는 방식으로 사용되고 있습니다.

→ **AI Cancer Segmentation, Web Sevice**의 기능은 아직 도입되지 않았습니다.



**ASAP | Automated Slide Analysis Platform**

------

ASAP은 QuPath와 동일하게 Annotation tool을 제공하고 있습니다. 해당 프로그램 또한 이미지를 업로드하여 진행하는 방식으로 서비스를 제공하고 있습니다.

→ **AI Cancer Segmentation, Web Sevice**의 기능은 아직 도입되지 않았습니다.



**Cytomine**

------

Cytomine은 저희가 **타겟**으로 삼는 서비스입니다. 웹 서버에서 병리 이미지들을 관리할 수 있으며 Annotation과 Cloud 서비스 등을 제공 중에 있습니다. 머신러닝에 대한 서비스도 제공 중에 있는 것으로 확인되고 있지만 해당 플랫폼은 서비스들을 **유료**로 제공하고 있습니다.

→ 제공되는 기능들은 모두 저희가 목표로 하는 서비스와 비슷하지만 서비스들이 **유료**로 제공되고 있는 플랫폼입니다.



### 🥅 Objective

------

- 병리이미지를 웹 서버에서 관리할 수 있도록 할 예정입니다. - Node.js
- 데이터베이스를 구축하여 이를 통해 병리이미지들을 보관할 예정입니다. - MySQL
- Annotation Tool을 Web에서 제공할 수 있도록 할 예정입니다.
- Zoom-in, Zoom-out의 기능을 제공할 예정입니다.
- Annotation에 도움이 될 수 있는 병리이미지 Cancer 예측 모델을 제공할 예정입니다. - Pytorch
- AI Cancer Segmentation의 속도를 더 빠르게 웹에 반영되도록 하는 방법을 찾고 있습니다. -Python

### 🕑 In-process

------

- MySQL 과 Node js를  통해 서버의 로그인 환경을 구축하였고 카카오 로그인도 추가하였습니다.
- 현재는 하드디스크를 통해 이미지들을 관리하고 있습니다. 하지만 곧 데이터베이스를 설계한뒤 이를 클라이언트와 연결하는 작업을 진행 할 것입니다.
- 현재 웹 기능으로는 이미지 파일을 업로드 다운로드 할 수 있으며 이미지에 대한 설명을 수정 할 수 있는 업데이트 기능 그리고 삭제 기능이 구현 되어 있습니다.
- Node js 와 python 을 이용해 Predict 버튼을 통해  AI Cancer Segmentation을 확인할 수 있도록 구현하였습니다. 현재 소모되는 시간은 20~30seconds 정도입니다. 이를 줄이는 방안을 모색중에 있습니다.
- Predict 버튼을 눌렀을 때, 소모되는 시간을 고려하여 Loading Bar를 추가하였습니다.
- 웹 서버에서 이미지의 Zoom-in, Zoom-out과 Annotation Tool을 제공하도록 구현하였습니다.



### 📖 Reference

------

- Daisuke Komura and Shumpei Ishikawa. "Machine Learning Methods for Histopath ological Image analysis" Computational and structural Biotechnology Journal, 16(2018), p.34-42
- Le Hou, Dimitris Samaras, Tashin M.Kurc, Yi Gao, James E. Davis et al. "Patch-based Convolutional Neural Network for Whole Slide Tissue Image Classification" In CVPR, 2016.

### 📖 Reference(etc.)

------

- [Python Multiprocessing](https://niceman.tistory.com/145)
- [Nodejs python-shell binary연동, 바이트 파일 주고 받기](https://asung123456.tistory.com/16)
- [Nodejs python-shell image 이미지 파일 Json으로 주고 받기](https://asung123456.tistory.com/15)
- [zoom 기능 opensource](https://github.com/jackmoore/zoom)