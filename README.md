# NLP code assignment3 XLNet

생성 일시: 2022년 11월 22일 오후 5:59
진행: 진행
태그: class

## XLNet: Generalized Autoregressive Pretraining for Language Understanding

### 1. Autoregressive (AR) vs Auto Encoding (AE)

- AR은 이전 token들을 보고 다음 toekn을 예측하는 문제를 푸는 문제 ex) GPT
    
    ![Untitled](NLP%20code%20assignment3%20XLNet%2063ecacd86b304b9c82ee9654073e7b05/Untitled.png)
    
    - AR은 방향성이 정해져야 하므로, 한쪽 방향의 정보만을 이용할  수 있기 때문에 양방향 문맥을 활용하기 어렵다는 단점이 존재
    
- AE는 주어진 input에 대해 그 input을 그대로 예측하는 문제를 풀고, Denoising Auto Encoder는 noise가 섞인 input을 원래의 input으로 예측하는 문제를 푼다. ex) BERT (MASK 토큰을 원래 input으로 복구)
    
    ![Untitled](NLP%20code%20assignment3%20XLNet%2063ecacd86b304b9c82ee9654073e7b05/Untitled%201.png)
    
    - 양방향 정보를 활용할 수 있다는 장점
    - 그러나 independent assumption을 적용하여 [MASK] token이 독립적으로 예측되기 때문에 [MASK] 사이의 dependency를 학습할 수 없다는 단점
    - pretrain 단계에서는 [MASK] 토큰이 있으나, fine-tuning 단계에서는 없기 때문에  불일치가 발생

### 2. XLNet

- AR과 AE의 장점을 취득하기 위해 제안된 모델로써 다음의 3가지를 주요 키워드로 새로운 방법을 제안
    - Permutation Language Modeling이라는 새로운 objective를 제안
    - 위 방법을 제안하기 위한 Target-Aware Representation 제안
    - Two-Stream Self-Attention 구조

### 2-1. Permutation Language Modeling Objective

![Untitled](NLP%20code%20assignment3%20XLNet%2063ecacd86b304b9c82ee9654073e7b05/Untitled%202.png)

- input sequence index 의 모든 permutation을 고려한 AR 방식을 이용
    - [x1,x2,x3,x4] 에 대해서
    - 4! = 24 개의 permutation, 즉 ZT=[[1,2,3,4],[1,2,4,3],[1,3,2,4]…[4,3,2,1]] 을 학습
    
    ![Untitled](NLP%20code%20assignment3%20XLNet%2063ecacd86b304b9c82ee9654073e7b05/Untitled%203.png)
    
- 각 token들은 원래 순서에 따라 positional encoding이 부여되기 때문에 토큰들의 상대적 위치를 구분할 수 있다.
- 양방향 context를 고려한 AR 모델링
- AE의 independent assumption이 없고, pre-training과 fine-tuning의 불일치도 없음

### 2-2. Target-Aware Representation for Transformer

- 아래의 예시와 같은 문제를 해결하기 위해,
    
    ![Untitled](NLP%20code%20assignment3%20XLNet%2063ecacd86b304b9c82ee9654073e7b05/Untitled%204.png)
    
- 이전의 context token들의 정보 뿐 아니라 target index의 position 정보도 함께 이용하는 target position-aware representation 제안
    
    ![Untitled](NLP%20code%20assignment3%20XLNet%2063ecacd86b304b9c82ee9654073e7b05/Untitled%205.png)
    

### 2-3. Two-Stream Self-Attention for Target Aware Representation

- target position 정보를 추가적으로 이용하는 g() 함수를 구성하기 위하여
- 한 개의 token 당 하나의 representation만 갖는 것이 아니라, 2개의 hidden representation을 이용하는 변형된 transformer 구조를 제안

1) Query Representation: 이전 시점 token들의 content와 현재 시점의 위치 정보를 이용하여 계산되는 representation 

- 수식
    
    ![Untitled](NLP%20code%20assignment3%20XLNet%2063ecacd86b304b9c82ee9654073e7b05/Untitled%206.png)
    

2) Context Representation: 현재 시점 및 이전 시점 token들의 content를 이용하여 계산되는 representation (표준 트랜스포머의 hidden state와 동일한 역할) 

- 수식
    
    ![Untitled](NLP%20code%20assignment3%20XLNet%2063ecacd86b304b9c82ee9654073e7b05/Untitled%207.png)
    

### 3. Incorporating Ideas from Transformer-XL

- 긴 문장에 대한 처리를 위해 Transformer-XL에서 사용된 2가지 테크닉 차용

### 3-1. Relative Positional Encoding

- multiple segments 에서의 absolute positional encoding에 대한 문제를 해결하기 위해
- input-level이 아닌 self-attention mechanism에서 relative positional encoding을 사용
- 이를 통해 상대적 위치 정보를 모델링

![Untitled](NLP%20code%20assignment3%20XLNet%2063ecacd86b304b9c82ee9654073e7b05/Untitled%208.png)

### 3-2. Segment Recurrence Mechanism

- 긴 문장에 대해서 여러 segment로 분리하고 이에 대해서 recurrent하게 모델링을 진행
- 첫 번째 segment에 대한 처리를 완료하고 각 layer m으로부터 얻어진 content representation을 캐싱하여 두 번째 segment에 대한 계산을 다음과 같이 진행함
    
    ![Untitled](NLP%20code%20assignment3%20XLNet%2063ecacd86b304b9c82ee9654073e7b05/Untitled%209.png)
    
- 이를 통해 과거 segment에 대한 factorization order를 고려하지 않고 memory의 caching과 reusing이 가능하다.
- 이를 통해서 sequential decoding의 속도를 빠르게 할 수 있고, 과거 segment에 계산된 것들에 대해서 input_ids를 넣지 않아도 된다.

---

### Finetuning XLNet with Pytorch

- spam classifiaciton 문제
- XLNetForSequenceClassification 모델을 활용하여 finetuning 진행