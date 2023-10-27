---
title: "[Background] Segmentation Evaluation Metrics"
date: 2023-10-24 22:50:00 +09:00
categories: [AI]
tags:
  [ Background ]
math: true
---

Segmentation task에서 평가 지표에 대해서 알아보도록 하겠습니다.

- **IoU(Intersection over Union)**

![fig1](/assets/img/segmentation_metric/fig1.png)

예측된 bounding box와 ground truth의 합집합과 교집합의 비율입니다.

![fig2](/assets/img/segmentation_metric/fig2.png)

0 ~ 1 사이의 값을 가지며, 1에 가까울수록 좋은 성능이고 0에 가까울수록 낮은 성능입니다.

object 각각의 IoU를 구해서 클래스별 평균을 구한 것이 mIoU입니다.

- **AP(Average Precision)**

precision-recall curve를 그리고 곡선 아래의 면적을 계산합니다.  

각 클래스 또는 카테고리에 대해 AP를 계산합니다. 이는 모든 클래스에 대한 AP 값을 평균낸 값입니다.

AP 값이 높을수록 모델이 더 많은 객체를 정확하게 감지하여 성능이 뛰어나다는 것을 의미합니다.

- **Panoptic Quality(PQ)**

Panoptic Segmentation에서 사용되는 평가 지표입니다. object detection과 semantic segmentation의 결과를 종합적으로 평가합니다.

Segmentation Quality(SQ)와 Recognition Quality(RQ)로 구성되어 있습니다.

SQ: Panoptic Segmentation 결과에서 Semantic Segmentation 부분을 평가하는 지표입니다. 정확하게 예측된 semantic 클래스와 해당 클래스에 대한 실제 픽셀 수(TP)를 기반으로 계산됩니다. 이는 패노픽 세그멘테이션 결과의 시맨틱 부분에서의 정확성을 나타냅니다.

하지만 이는 TP에 대해서만 고려하고 있으므로 bad prediction에 대한 것이 없습니다.

따라서 precision, recall 등을 잘 고려하여 모델 성능에 대해 더 잘 평가하기 위해서 RQ가 존재합니다.

RQ: Panoptic Segmentation 결과에서 Instance Segmentation 부분을 평가하는 지표입니다. 정확하게 예측된 객체 인스턴스와 해당 객체 인스턴스의 실제 픽셀 수(TP)를 기반으로 계산됩니다. 이는 패노픽 세그멘테이션 결과의 객체 인스턴스 인식 정확성을 나타냅니다.

- **DICE Coefficient**

두 집합의 유사성을 측정하는 통계적인 지표 중 하나입니다. 영역의 중첩을 평가하여 두 영역의 유사성을 0부터 1까지의 값으로 표현합니다.

![fig4](/assets/img/segmentation_metric/fig4.png)

DICE Coefficient는 세그멘테이션 모델의 성능을 평가하거나 두 세그멘테이션 결과 간의 유사성을 비교하는 데 사용됩니다. 0 ~ 1 사이의 값을 가지며, 1에 가까울수록 두 세그멘테이션 결과 간의 유사성이 더 높음을 나타냅니다.

![fig3](/assets/img/segmentation_metric/fig3.png)

