---
title: "[Background] Inductive Bias"
date: 2023-09-30 11:50:00 +09:00
categories: [AI]
tags:
  [ Background ]
math: true
---

논문을 공부하면서 inductive bias라는 용어를 자주 볼 수 있는데요

오늘은 inductive bias에 대해서 알아보겠습니다.

## Inductive Bias란?

모델이 데이터로부터 일반화를 추론하는 데 도움을 주는 미리 종의된 가정이나 제한 사항을 의미하는데요.

일반적으로 모델이 갖는 generalization problem은 모델이 brittle(불안정)하다는 것과 spurious(겉으로만 그럴싸한)하다는 것이 있습니다.

> **Models are Brittle**: 아무리 같은 의미의 데이터라도 조금만 바뀌면 모델이 망가진다.  
> **Models are Spurious**: 데이터 본연의 의미를 학습하는 것이 아닌 결과와 편향을 학습하게 됩니다.

모델이 주어진 데이터에 대해 잘 일반화되었는지, 혹은 주어진 데이터에만 맞는 것인지 모르기 때문에 발생하는 문제입니다.  
이러한 문제를 해결하기 위해 Inductive Bias가 등장했습니다.  
**Inductive Bias란, 학습 시에는 만나보지 않았던 상황에 대하여 정확한 예측을 하기 위해 사용하는 추가적인 가정(additional assumptions)**을 의미합니다.  

모델이 학습하는 과정에서 제한된 학습 데이터가 주어지게 됩니다. 학습 데이터 이외의 다른 데이터에 대해서도 일반화하기 위해 Inductive Bias가 존재합니다.

몇 가지 Induvtive Bias 예시입니다.

>- Translation invariance: 어떠한 사물이 들어 있는 이미지에서 사물의 위치가 바뀌어도 해당 사물을 인식할 수 있습니다.
>
>- Translation equivariance: 어떠한 사물이 들어 있는 이미지에서 사물의 위치가 바뀌면 CNN과 같은 연산의 activation 위치 또한 바뀌게 됩니다.
>
>- Maximum conditional independence: 가설이 베이지안 프레임워크에 캐스팅될 수 있다면 조건부 독립성을 극대화합니다.
>
>- Minimum cross-validation error: 가설 중에서 선택하려고 할 때 교차 검증 오차가 가장 낮은 가설을 선택합니다.
>
>- Maximum margin: 두 클래스 사이에 경계를 그릴 때 경계 너비를 최대화합니다.
>
>- Minimum description length: 가설을 구성할 때 가설의 설명 길이를 최소화합니다. 이는 더 간단한 가설은 더 사실일 가능성이 높다는 가정을 기반으로 하고 있습니다.
>
>- Minimum features: 특정 피쳐가 유용하다는 근거가 없는 한 삭제해야 합니다.
>
>- Nearest neighbors: 특징 공간에 있는 작은 이웃의 경우 대부분이 동일한 클래스에 속한다고 가정합니다.


## 딥러닝에서의 Induvtive Bias

딥러닝에서 흔히 쌓는 레이어의 구성은 일종의 Relational Inductive Bias(관계 귀납적 편향), 즉 hierarchical processing(계층적 처리)를 제공합니다.  
딥러닝 레이어의 종류에 따라 추가적인 관계 유도 편향이 부과됩니다.  

![fig7](/assets/img/inductive_bias/fig7.png)

## Inductive Biases on CNN, RNN, GNN

Image Classification, Object Detection 등 이미지를 다루는 모델들은 CNN을 사용합니다. CNN이 이미지를 다루기에 적합한 Inductive Bias를 갖고 있기 때문입니다. 

FCN(Fully Connected Neural Network)은 가장 일반적인 블록 형태로, weight와 bias로 각 층의 모든 요소들이 서로 연결되어 있습니다. 즉, 모든 입력의 요소가 모든 출력 요소에 영향을 미치기 때문에, Inductive Bias가 매우 약합니다.

반면에, CNN(Convolutional Neural Network)은 FCN과 다르게 Locality & Translation Invariance의 Inductive Bias를 갖습니다.

Locality는 입력에서 각 Entities 간의 관계가 서로 가까운 요소들에 존재한다는 것을 의미합니다. Translation Invariance란 입력 데이터 내에서 객체가 이동하더라도 모델은 그 객체를 동일하게 인식하고 처리할 수 있어야 함을 의미합니다.

CNN이 공간의 개념을 사용한다면, RNN은 시간의 개념을 사용합니다.   
RNN에서는 CNN의 Locality & Translation Invariance와 유사한 개념으로 Sequential & Temporal Invariance의 Inductive Bias를 갖습니다.  
Sequential이란 입력이 시계열의 특징을 갖는다고 가정하며, Temporal Invariance는 입력의 순서와 출력의 순서가 동일하다는 것을 의미합니다.

GNN(Graph Neural Network)은 이러한 개념을 그래프로 가져간 것으로, Permutational Invariance의 Inductive Bias를 갖습니다.



## 마무리

Inductive Bias가 강할수록, 적은 양의 훈련 데이터로도 모델이 효과적으로 학습하고 좋은 성능을 낼 수 있지만 그만큼 가정이 강하게 들어간 것이라 좋은 것만임은 아닙니다. 이는 편향-분산 트레이드 오프 개념과 유사합니다.


[출처](https://velog.io/@euisuk-chung/Inductive-Bias%EB%9E%80)