---
title: "Image Segmentation"
date: 2023-10-24 21:50:00 +09:00
categories: [AI]
tags:
  [ Background ]
math: true
---

이번에는 Panoptic Segmentation에 대해서 알아보겠습니다.

Panoptic Segmentation은 Image Segmentation task 종류 중의 하나입니다.

먼저 Image Segmentation의 종류에 대해 간단하게 정리해봅시다.

Image Segmentation은 이미지를 몇 개의 object로 분할하는 task입니다.

![fig1](/assets/img/image_segmentation/fig1.png)

Semantic Segmentation은 이미지의 모든 요소에 대해 클래스 레이블을 예측합니다. 동일한 클래스 라벨에 해당하는 객체에 대해서 구분하지 않습니다.
Semantic Segmentation 모델로는 DeepLab, DeepLabV3, FastFCN, Transformer-based models가 있습니다.

Instance Segmentation은 이미지의 모든 object에 대해서 클래스 레이블을 예측하고, 개별 object에 대해 구분을 합니다. 즉, 이미지의 각 object에 고유한 클래스 레이블을 할당하고 각 object의 경계도 식별합니다.  
Semantic Segmentation과의 차이점은 겹쳐진 물체에 대해서 각각 검출한다는 점입니다. 또한, 하늘이나 도로 등 정해진 형태가 없는 경우 클래스 레이블을 부여하지 않습니다.  
Instance Segmentation은 이미지의 객체가 모양이나 컨텍스트에 따라 이미지의 개별 object를 식별하고 세분화할 수 있게 됩니다.

Instnace Segmentation은 두 가지 방식으로 수행됩니다.  

- Bottom-up 방식: 이미지의 개별 픽셀을 감지하고, 이러한 픽셀을 그룹화하여 object를 형성합니다.

- Top-down 방식: 이미지의 전체 장면을 감지하고, 개별 object를 식별한 뒤 세분화합니다.

Panoptic Segmentation은 Semantic Segmentation과 Instance Segmentation을 결합한 형태로 이미지 안의 모든 픽셀을 물체와 배경으로 분할하고, 각 물체에 대해 클래스를 부여하는 작업입니다. Semantic Segmentation을 통해 이미지의 object를 식별할 수 있으며, Instance Segmentation을 통해 이미지의 개별 object를 식별하고 세분화할 수 있습니다. 이 작업은 Stuff과 Thing에 대한 정보를 제공합니다.

- Stuff: 물체가 아닌 배경과 비슷한 패턴, 질감 또는 색상을 가지는 영역입니다. 예를 들어, 하늘, 도로, 풀, 벽 등이 해당합니다. Stuff는 Semantic Segmentation을 통해 식별됩니다.

- Thing: 개별 물체 또는 객체를 나타내는 영역입니다. 이 영역은 물체의 클래스와 인스턴스 정보를 가지며, 객체 간의 구분도 수행됩니다. 예를 들어, 사람, 자동차, 강아지 등이 해당되며, Instance Segmentation을 통해 식별됩니다.

Panoptic Segmentation에서는 PQ(Panoptic Quality)라는 성능 지표를 통해 segmentation model을 평가합니다.

