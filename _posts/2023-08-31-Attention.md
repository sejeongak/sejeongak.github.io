---
title: Attention Is All You Need
date: 2023-08-31 16:50:00 +09:00
categories: [AI, 논문]
tags:
  [ NLP ]
math: true
---

안녕하세요

이번에 다룰 논문은 "Attention Is All You Need" 논문입니다.

정말 유명한 논문이죠!
논문의 제목조차 한번 들으면 잊을 수 없는 임팩트를 가지고 있습니다.
이 논문은 BERT, GPT 등의 기반이 되어 NLP 분야에 엄청난 영향력을 끼친 논문입니다.

처음으로 RNN, CNN을 완전히 배제하고 전적으로 어텐션 메커니즘에 기반한 Transformer 아키텍처를 제안한 논문인데요.  
이는 시퀀스 데이터를 병렬적으로 처리하기 때문에 훨씬 더 빠른 학습이 가능하게 됩니다.

WMT 2014 영어-독일어 번역 작업에서 기존의 최고 성능 모델보다 우수한 결과를 보였고, WMT 2014 영어-프랑스어 번역 작업에서는 새로운 단일 모델 SOTA를 달성했습니다.  
심지어 이전 최고 성능 모델들보다 훨씬 적은 비용으로 학습되었다고 합니다!

그럼 논문에 대해서 더 자세하게 리뷰해보도록 하겠습니다.

# 1. Introduction

- 기존 Sequence Model의 한계

기존의 Sequence Model은 입력 시퀀스를 한 번에 처리하고 고정된 길이의 벡터를 생성했습니다.
따라서 입력 시퀀스가 길어질수록 서로 멀리 떨어진 문장에 대한 정보가 줄어들기 때문에 성능이 떨어집니다.
또한 병렬적으로 처리가 어렵기 때문에 입력 시퀀스의 길이가 길어질수록 연산 속도가 저하됩니다.

- Attention Mechanism

이런 한계를 극복하기 위해 Attention Mechanism이 등장하였습니다. 
recurrent network를 배제하고도 입력과 출력 간의 전역 의존성을 모델링할 수 있게 Attention Mechanism만을 사용한 Transformer Architecture를 제안했습니다.

# 2. Background

선행 연구들은 입력과 출력 간의 거리가 멀수록 의존성을 학습하기 어렵다는 한계점을 가지고 있습니다.

Transformer에서는 연산량을 감소시켰으나, 이로 인해 attention-weighted positions가 평균화되는 것으로 인해 실제 성능이 감소하는 문제가 발생합니다.

>💡
>"attention-weighted positions"는 attention mechanism에서 각 위치 또는 토큰에 할당되는 가중치를 의미합니다. 어텐션 메커니즘은 입력 시퀀스의 각 원소가 다른 원소들과 얼마나 관련 있는지를 계산하여 가중치를 생성합니다. 가중치가 평균화되면, 모든 위치가 비슷한 중요도로 처리될 수 있습니다. 결과적으로 모델은 입력 시퀀스 내의 중요한 관계나 패턴을 충분히 학습하지 못할 수 있습니다.

논문에서는 이러한 문제를 Multi-Head Attention으로 상쇄시킴으로써 극복합니다.
Multi-Head Attention에 대해서는 다음 절에서 설명해드리겠습니다.


Self-Attention은 시퀀스의 서로 다른 위치 간의 의존성을 학습합니다. 시퀀스에서 각 원소가 다른 원소들과 어떤 정도로 연관이 있는지를 계산합니다.

Transformer는 RNN이나 CNN을 사용하지 않고 전적으로 Self-Attention만을 활용하여 입력과 출력의 표현을 계산하는 최초의 변환 모델입니다.

# 3. Model Architecture

대부분의 경쟁력 있는 신경망 기반 시퀀스 변환 모델은 인코더-디코더 구조를 가지고 있습니다.  
Transformer 역시 encoder-decoder 구조를 가집니다.  
트랜스포머의 인코더는 입력 시퀀스 $ ( x_1, ..., x_n) $ 를 연속적인 시퀀스 $ z = (z_1, ..., z_n) $ 로 매핑합니다.
z가 주어지면, 디코더는 한 번에 하나씩 출력 시퀀스 $ (y_1, ..., y_m) $를 생성합니다.
각 단계에서 모델은 이전에 생성한 요소를 추가 입력으로 사용하여 다음 요소를 생성하는 auto-regressive 방식을 사용합니다.

![model_architecture1](/assets/img/Attention/model_architecture.png)

다음 그림은 Transformer의 전반적인 model architecture입니다.
인코더와 디코더의 각 layer는 stacked self-attention과 point-wise, fully connected layers로 구성되어 있습니다.

## 3.1 Encoder and Decoder Stacks

- Encoder

인코더는 6개의 동일한 layer로 구성되어 있습니다.
각 레이어는 2개의 sub layer(multi-head self-attention layer와 position-wise fully connected feed-forward layer)로 이루어져 있습니다.

각각의 sub layer마다 residual connection을 사용하고, layer normalization을 적용했습니다.
즉, 각 sub layer의 출력은 LayerNorm(x + sub layer(x))입니다.

이러한 residual connection을 용이하게 하기 위해 모델의 모든 sub layer 및 embedding layer는 출력 차원이 512인 결과를 생성합니다.

- Decoder

디코더 역시 6개의 동일한 layer로 구성되어 있습니다.
각 레이어는 Encoder layer의 두 sub-layer 외의 인코더의 출력에 대해 Multi-Head Attention을 수행하는 레이어를 추가로 가집니다.

>💡Multi-Head Attention vs Self-Attention
> - Multi-Head Attention: 인코더의 출력 시퀀스와 현재 쿼리 벡터 사이에서 내적 연산을 수행하여 어텐션 가중치를 계산
> - Self-Attention: 이전 디코더 레이어에서 생성된 출력 시퀀스와 현재 쿼리 벡터 사이에서 내적 연산을 수행하여 어텐선 가중치를 계산

인코더와 마찬가지로, residual connection과 layer normalization을 사용합니다.
주목해야 할 부분은 다음 위치에 있는 데이터에는 attention하지 못하도록 decoder stack의 self-attention sub layer를 수정했습니다.
디코더는 입력 시퀀스를 순차적으로 처리하므로, 미래의 위치에 해당하는 정보는 아직 알 수 없기 때문입니다.
이를 통해 i 번째 위치에 대한 예측은 i 이전 위치의 출력에만 의존할 수 있도록 합니다.

## 3.2 Attention

Attention function은 query와 key-value 쌍을 output에 매핑합니다.
여기서 query, key, value는 모두 벡터입니다.

output은 value들의 weighted sum으로 계산되며, 각 value에 할당되는 weight는 query와 대응되는 key의 유사도에 의해 계산됩니다.

![fig2](/assets/img/Attention/fig2.png)

왼쪽 그림은 Scaled Dot-Product Attention이고, 오른쪽 그림은 여러 attention layer가 병렬적으로 실행되는 Multi-Head Attention입니다.

### 3.2.1 Scaled Dot-Product Attention

input은 query($ d_k$ ), key($ d_k $ ), value($ d_v $)로 구성됩니다.
query를 모든 key에 대해 dot product를 계산하고, 각각을 $ \sqrt {d_k} $ 로 나눈 후, softmax 함수를 적용하여 value와 계산할 가중치를 얻습니다.

실제로는, 여러 개의 query를 동시에 계산하기 위해 행렬 Q에 함께 묶어서 계산한다고 합니다. 또한 key와 value도 행렬 K와 V로 함께 묶어 계산합니다.

다음과 같이 계산식으로 나타낼 수 있습니다.

![scaled_dot_product](/assets/img/Attention/scaled_dot_product.png)

흔히 쓰이는 attention function으로 additive attention, dot-product attention 방식이 있습니다.

- Additive Attention

Additive attention은 single hidden layer를 가진 feed-forward network를 사용하여 compatibility function을 계산합니다.

선형 연산으로, 입력 벡터와 가중치 벡터를 연결하여 점수를 계산합니다.

유연하고 여러가지 형태의 attention score를 계산할 수 있다는 장점이 있지만, 연산량이 많아 학습이 느리고 복잡한 모델에서 사용하기 어렵다는 단점이 있습니다.

- Dot-Product Attention

논문의 알고리즘과 동일하지만, $ {1 \over \sqrt {d_k}} $ 으로 스케일링한다는 점이 다릅니다.

내적 연산으로, 인코더 출력과 디코더 입력 벡터를 내적하여 점수를 계산합니다.

연산이 적고, 학습이 빠르며 다양한 분야에서 좋은 성능을 낼 수 있지만, 입력 벡터 차원 수가 클 경우 overflow나 underflow 문제가 발생할 수 있습니다.

$ d_k $가 작은 경우엔 두 방식이 유사한 성능을 가지지만, $ d_k $가 클 경우 additive attention의 성능이 스케일링이 없는 dot-product attention보다 더 좋습니다.

따라서 dot-product의 스케일링 값을 $ \sqrt {d_k} $로 설정하여 보완합니다.

>💡$ \sqrt {d_k} $ 로 나누는 이유
> 내적값의 크기는 두 벡터의 차원 수에 따라 달라집니다.
> 내적값의 크기가 너무 크거나 작아지면 softmax 계산에서 overflow, underflow 등의 문제가 발생할 수 있습니다.
> 따라서 스케일링을 통해 이러한 문제점을 방지하고, 안정적으로 학습할 수 있게 합니다.

### 3.2.2 Multi-Head Attention

$ d_{model} $ 차원의 key, value, query에 단일 어텐션 함수를 수행하는 대신, 본 논문은 서로 다른, 학습된 linear projection을 사용하여 각각 $ d_query $, key, value를 h번의 서로 다른, learned linear projection으로 $ d_k, d_k, d_v $ 차원에 linear하게 projection하는 것이 더 효과적인 것을 발견했습니다.

projection된 버전의 query, key, value 값 각각이 병렬적으로 attention function을 수행하여 $ d_v $ 차원의 output value를 얻습니다.

이들을 concat한 후, 다시 projection하여 최종 결과값을 생성합니다.

![multi-head-attention](/assets/img/Attention/multi-head-attention.png)

- projection parameter matrices

$$ W^Q_i ∈ \mathbb {R}^{d_{model}×d_k} $$

$$ W^Q_i ∈ \mathbb {R}^{d_{model}×d_k} $$

$$ W^V_i ∈ \mathbb R^{d_{model} ×d_v} $$

$$ W^O ∈ \mathbb R^{hd_v×d_{model}} $$

![fig3](/assets/img/Attention/fig3.png)

본 논문에서는 8개의 head를 사용했습니다.
각각의 attention layer에서는 $ d_{model} $ 차원의 입력값을 h개의 다른 head로 나누어 처리합니다.

$$ d_k = d_v = d_{model}/h = 64 $$


각 head마다 차원을 줄이기 때문에, 총 계산 비용은 전체 차원을 가진 single-head attention과 유사합니다.


### 3.2.3 Applications of Attention in our Model

Transformer는 세 가지 방법으로 multi-head attention을 사용합니다.

- "encoder-decoder attention" layer  
  query는 이전 디코더 layer에서 옵니다. key와 value는 인코더의 output에서 옵니다. 이를 통해 디코더의 각 위치가 입력 시퀀스의 모든 위치를 참조 가능하게 합니다.

- "encoder"  
  인코더에는 self-attention layer가 포함되어 있습니다. self-attention layer에서는 모든 key, value, query를 같은 위치(인코더의 이전 layer의 output)에서 가져옵니다. 인코더의 각 position은 인코더의 이전 layer의 모든 position을 참조할 수 있습니다.

- "decoder"  
  디코더 역시 self-attention layer를 포함하고 있습니다. 인코더와 달리, 디코더의 self-attention layer에서 디코더 내의 각 position은 해당 position을 포함한 디코더의 이전 position에 대해서만 참조할 수 있습니다.
  auto-regressive 속성을 보존하기 위해 디코더의 left forward information flow를 방지해야 합니다. 이를 scaled-dot product attention 내에서 구현하며, softmax의 입력에서 illegal connection에 해당하는 모든 값을 마스킹 처리($ - \inf $ 로 설정)합니다.

### 3.3 Position-wise Feed-Forward Networks

인코더와 디코더의 각 레이어는 attention sub-layer 외에도 각 위치에 대해 별도로 동일하게 적용되는 fully connected feed-forward network가 포함되어 있습니다. 
이 네트워크는 두 개의 선형 변환으로 이루어져 있으며 중간에 ReLU 활성화 함수가 적용됩니다.
다음 인코더/디코더 레이어에 전달하기 전에 각 위치에서의 토큰 표현을 더 정확하게 만들어줍니다.

![fig4](/assets/img/Attention/fig4.png)

이 선형 변환은 모든 위치에 대해 동일하게 적용되지만, 각 레이어 간에는 다른 매개변수(weight와 bias)를 사용합니다.
이것은 마치 커널 크기가 1인 컨볼루션 연산을 수행하는 것과 유사합니다.
즉, 각 위치의 입력을 독립적으로 처리하면서 가중치가 공유됩니다.

이는 각 레이어가 서로 다른 특성을 학습하고 다양한 정보를 캡처할 수 있게 하여 모델 성능을 향상시킵니다.

입력과 출력의 차원은 $ d_{model} = 512 $ 이며, 내부 레이어는 $ d_{ff} = 2048 $ 의 차원을 가집니다.
이는 경험적으로 설정한 값입니다.

### 3.4 Embeddings and Softmax

기존 시퀀스 변환 모델들과 마찬가지로, 입력 토큰과 출력 토큰을 학습된 embedding layer를 사용하여 $ d_{model} $ 차원의 벡터로 변환합니다.
또한 디코더 출력을 예측된 다음 토큰 확률로 변환하기 위해 선형 변환과 softmax 함수를 사용합니다.  
이 모델에서는 두 개의 embedding layer와 pre-softmax 선형 변환 간에 동일한 weight matrix를 공유합니다.  
이를 통해 입력과 출력 간에 더 많은 정보를 공유하며, 학습 파라미터 수가 감소하여 모델의 학습 속도를 향상시킬 수 있습니다.  
embedding layer에서는 이 가중치 행렬을 $ \sqrt d_{model} $ 로 스케일링하여 사용합니다.

$ \sqrt d_{model} $로 스케일링을 함으로써 어텐션 메커니즘에서 가중치를 조절하여 모델의 안정적인 학습을 돕는 효과를 얻을 수 있습니다.

### 3.5 Positional Encoding

Transformer는 recurrence, convolution을 사용하지 않기 때문에 sequence의 순서를 활용하려면 sequence의 상대적인, 절대적인 위치 정보가 필요합니다.  
이를 위해 인코더와 디코더 스택의 input embedding에 "positional encoding"을 추가합니다.

모든 위치 값은 시퀀스의 길이나 input에 관계없이 동일한 식별자를 가져 시퀀스가 바뀌더라도 위치 임베딩은 동일하게 유지됩니다.

위치 값이 너무 커져버리면, 단어간의 상관관계 및 의미를 유추할 수 있는 의미 정보 값이 상대적으로 작아지게 되어 학습이 제대로 되지 않기 때문에 적절한 값으로 조절하는 것이 중요합니다.

positional encoding은 embedding과 동일한 $ d_{model} $ 차원을 가지므로 서로 더할 수 있습니다.  
이는 단어 의미 정보와 위치 정보 간의 균형을 잘 맞출 수 있지만, 반대로 정보가 뒤섞이는 문제가 발생할 수 있습니다.

positional encoding은 학습 가능한 방법과 고정된 방법 등 다양한 선택지가 있습니다.

이 논문에서는 서로 다른 frequency를 갖는 sin 함수와 cosine 함수를 사용합니다.  
위치 벡터 값이 같아지는 문제를 해결하기 위해 다양한 주기의 sin & cosine 함수를 동시에 사용하여 하나의 위치 벡터를 여러 차원으로 표현했습니다.
항상 -1 ~ 1 사이의 값을 얻습니다.

더 긴 문장에 대해서도 상대적인 positional encoding이 가능합니다.

![fig5](/assets/img/Attention/fig5.png)

그림에서 pos는 position, i는 dimension을 의미합니다.


이 함수를 선택한 이유는 어떤 고정된 offset k에 대해서도, $ PE_{pos + k} $ 를 $ PE_{pos} $ 의 선형 함수로 나타낼 수 있어 모델이 상대적인 position에 대한 attention을 쉽게 학습할 수 있을 것이라 생각했기 때문입니다.

또한 학습 가능한 위치 임베딩을 사용하는 실험도 수행했습니다.
두 버전이 거의 동일한 결과를 산출하는 것을 발견했습니다. (표 3의 (E) 행 참조)

본 논문에서는 시퀀스의 길이가 더 긴 경우에도 모델이 추론할 수 있게 사인 함수 버전을 선택했습니다.  
사인 함수 버전은 -1 ~ 1 사이를 주기적으로 반복하기에 긴 문장의 시퀀스에서도 위치 벡터 값의 차가 작지 않게 됩니다.

# 4. Why Self-Attention

이 장에서는 본 논문이 왜 Self-Attention을 사용했는지에 대한 내용이 담겨있습니다.

본 논문은 세 가지 측면으로 recurrence, convolution과 비교하여 Self-Attention을 사용했습니다.

  1. 레이어 당 총 계산 복잡도  
  각 레이어에서 수행되는 계산 작업의 양으로 모델이 처리하는 작업의 양을 의미합니다.
  2. 연속적인 연산이 최소로 필요한 병렬화 가능한 계산 양  
  계산을 얼마나 효율적으로 병렬로 처리할 수 있는지를 나타내며, 더 적은 순차적인 단계가 필요하면 모델을 더 효율적으로 병렬화하여 빠른 계산이 가능하게 됩니다.
  3. 네트워크 내에서 장거리 의존성 간의 경로 길이  
  장거리 의존성을 학습하는 것은 많은 시퀀스 변환 작업에서 중요한 도전 과제 중 하나입니다. 이러한 종속성을 학습하는 데 영향을 미치는 주요 요인 중 하나는 신호가 네트워크 내에서 진행하는 경로의 길이입니다. 입력 및 출력 시퀀스의 모든 위치 조합 간의 경로가 짧을수록 장거리 의존성을 학습하기 더 쉽습니다. 따라서 서로 다른 레이어 유형으로 구성된 네트워크에서 어떤 두 입력 및 출력 위치 사이의 최대 경로 길이를 비교합니다.

![table1](/assets/img/Attention/table1.png)

Self-attention 레이어의 경우, 모든 위치의 쿼리와 모든 위치의 키 사이의 어텐션 스코어를 계산합니다. $ O(n^2) $개의 어텐션 스코어를 계산하므로 $ O(n^2 * d) $ 의 계산 복잡성이 발생합니다.

Recurrent 레이어는 각 위치의 input을 이전 state와 결합하여 새로운 state를 생성합니다.  
input과 state의 각 차원마다 내적 연산 및 행렬곱이 필요하므로 $ O(d^2) $ 복잡성이 각 위치에서 발생합니다. 따라서 recurrent 레이어의 계산 복잡성은 $ O(n*d^2) $가 됩니다.

Convolutional 레이어는 커널이 입력 시퀀스를 슬라이딩하면서 내적 연산을 수행합니다. 따라서 계산 복잡성은 $ O(k * n * d^2) $ 가 됩니다.

계산 복잡성 측면에서, 시퀀스 길이 n이 표현 차원 d보다 작은 경우 self-attention 레이어는 순환 레이어보다 빠릅니다.    
매우 긴 시퀀스를 다루는 작업의 계산 성능을 향상시키기 위해서는 self-attention을 각 출력 위치 주변에 중심을 둔 입력 시퀀스의 크기가 r인 부분만 고려하도록 제한할 수 있습니다. 이렇게 하면 최대 경로 길이가 $ O(n/r) $ 로 증가합니다.

$ k < n $ 을 가진 하나의 컨볼루션 레이어는 입력과 출력 위치의 모든 쌍을 연결하지 않습니다. 이렇게 하려면 연속 커널의 경우 $ O(n/k) $의 컨볼루션 레이어 스택이 필요하며, 팽창된(dilated) 컨볼루션의 경우 $ O( \log_k{(n)}) $ 이 필요합니다. 이는 네트워크 내의 어떤 두 위치 사이의 가장 긴 경로의 길이를 증가시킵니다. 컨볼루션 레이어는 일반적으로 순환 레이어보다 k의 배수로 비용이 더 많이 듭니다. 그러나 분리형 컨볼루션(separable convolutions)은 복잡성을 상당히 줄여주어 $ O(k \cdot n \cdot d + n \cdot d^2) $ 로 감소됩니다. 그러나 $ k = n $ 인 경우에도 분리형 컨볼루션의 복잡성은 본 논문에서 채택한 self-attention layer와 point-wise feed-forward layer의 조합과 동일합니다.

self-attention 구조는 입력 시퀀스의 모든 단어들간의 관계를 고려하기 때문에, 기존의 모델에 비해 더욱 강력한 모델링 능력을 가지며, 번역, 요약, 질의응답 등 다양한 자연어 처리 task에사 우수한 성능을 보입니다.  
따라서 self-attention을 이용한 모델링이 기존의 sequence-to-sequence 모델에 비해 더욱 우수한 성능을 보입니다.

부수적인 이점으로, self-attention은 더 해석 가능한 모델을 제공할 수 있습니다. 개별 어텐션 헤드는 명확하게 서로 다른 작업을 수행하는 것으로 보이며, 많은 헤드가 문장의 구문 및 의미 구조와 관련된 행동을 나타내는 것으로 보입니다.


# 5. Training

## 5.1 Training Data and Batching

영어-독일어 번역 task에서는 4.5M 문장 쌍으로 구성된 WMT 2014 영어-독일어 데이터셋을 사용했습니다.
문장은 byte-pair encoding을 사용하여 인코딩되었으며, 이 방법은 약 37,000개의 토큰으로 구성된 공유 소스 대상 어휘를 가지고 있습니다.

영어-프랑스어 번역 task에서는 WMT 2014 영어-프랑스어 데이터셋을 사용했습니다. 이 데이터셋은 36M 문장, 32,000개의 word-piece vocabulary로 분리된 토큰들로 구성되어 있습니다.
문장 쌍은 근사적인 시퀀스 길이에 따라 함께 배치되었습니다. 각 학습 배치는 약 25,000개의 source token과 25,000개의 target token을 포함하는 문장 쌍 집합을 포함하고 있습니다.

## 5.2 Hardware and Schedule

- 8 NVIDIA P100 GPU
- base model -> 12시간 동안 100,000 step 학습(step당 0.4초)
- big model -> 3.5일 동안 300,000 step 학습(step당 1.0초)

## 5.3 Optimizer

- Adam optimizer 사용( $ \beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-9} $ )

![fig6](/assets/img/Attention/fig6.png)

- 처음 warmup_steps번의 학습 스텝에서는 learning rate를 선형적으로 증가시키고, 이후에는 학습 스텝 수의 제곱근의 역수에 비례하여 감소시키는 공식. warmup_steps의 값은 4,000으로 설정

## 5.4 Regularization

학습 중 세 가지 종류의 정규화를 적용했습니다.

- Residual Dropout
  - 각 sub-layer의 output에 dropout을 적용
  - encoder, decoder layer에서 임베딩과 positional 인코딩의 합에 dropout 적용
  - $ P_{drop} = 0.1 $

- Label Smoothing
  - 학습 중에 label smoothing 적용 ( $ \epsilon_{ls} = 0.1 $ )
  - 이는 모델이 더 확실하지 않은 결과를 내도록 학습되므로, perplexity를 저하시키지만, 정확도와 BLEU 점수 향상

![table2](/assets/img/Attention/table2.png)

# 6. Results

## 6.1 Machine Translation

WMT 2014 영어-독일어 번역 task에서 본 논문의 big model은 기존 sota BLEU score를 2.0 이상 개선하여 28.4의 새로운 sota BLEU score를 달성했습니다.
심지어 base model도 이전에 발표된 모든 모델과 앙상블을 능가하며, 경쟁 모델 중 어떤 것보다도 더 낮은 훈련 비용으로 학습이 진행되었습니다.

WMT 2014 영어-프랑스어 번역 task에서 본 논문의 big model은 41.0의 BLEU score를 달성하여 이전에 발표된 모든 단일 모델을 능가하며, 이전 최고 성능 모델의 1/4 미만의 훈련 비용으로 달성했습니다.
dropout rate $ P_{drop} = 0.1 $ 을 사용했습니다.

베이스 모델의 경우, 10분 간격으로 작성된 마지막 5개 체크포인트를 평균화하여 얻은 단일 모델을 사용했습니다.
대규모 모델의 경우, 마지막 20개 체크포인트를 평균화했습니다.
beam search를 사용하며 beam 크기는 4, 길이 페널티 $ \alpha $ 는 0.6으로 설정했습니다. 추론 중에 최대 출력 길이는 입력 길이 +50으로 설정되었으나 가능한 경우 조기 종료되었습니다.

표 2는 다른 모델 구조와 비교하여 번역 품질과 훈련 비용을 비교합니다.
모델을 훈련하는 데 사용된 부동 소수점 연산 횟수를 추정하기 위해 훈련 시간, 사용된 GPU 수 및 각 GPU의 지속 단정밀도 부동 소수점 용량을 곱하여 계산합니다.


## 6.2 Model Variations

![table3](/assets/img/Attention/table3.png)

Transformer의 다른 요소들의 중요성을 평가하기 위해, 베이스 모델을 다른 방식으로 변형하면서 성능 변화를 측정했습니다.

Table 3의 (A) 행에서는, 계산 양을 일정하게 유지하면서 attention head 수와 key, value의 차원 수를 변화시켰습니다.
단일 헤드 어텐션은 최적 설정과 비교했을 때 0.9 BLEU 낮은 성능을 보이지만, 너무 많은 어텐션 헤드는 성능이 떨어집니다.

Table 3의 (B) 행에서는, key의 차원 크기를 줄이면 모델의 품질이 하락한다는 것을 보여줍니다.
이는 compatibility 결정이 쉽지 않으며, 단순한 dot product 함수보다 더 복잡한 compatibility 함수가 유익할 수 있음을 시사합니다.

(C)와 (D) 행에서는, 대규모 모델이 더 나은 성능을 보이며, dropout은 과적합을 방지하는 데 매우 유용하다는 것을 보여줍니다.

마지막으로 (E) 행에서는, sin 함수를 사용한 positional encoding 대신 학습 가능한 positional embedding을 사용하였을 때 거의 동일한 결과를 얻을 수 있다는 것을 보여줍니다.

## 6.3 English Constituency Parsing

![table4](/assets/img/Attention/table4.png)

Transformer 모델이 다른 task에도 일반화될 수 있는지 평가하기 위해 영어 구문 분석 작업에서 실험을 수행했습니다.

Table 4는 모델이 task 특화 튜닝없이도 상당히 잘 수행되어 순환 신경망 문법을 제외한 이전의 모든 모델을 능가한다는 것을 보여줍니다.

RNN 기반의 seq2seq 모델과는 달리 Transformer 모델은 WSJ 학습 세트에서 학습할 때도 BerkeleyParser보다 우수한 결과를 얻습니다.

# 7. Conclusion

Transformer는 recurrent layer를 multi-head self-attention으로 완전히 대체한 첫 번째 sequence 변환 모델입니다.

WMT 2014 영어-독일어 및 WMT 2014 영어-프랑스어 번역 작업에서 SOTA를 달성했습니다.








