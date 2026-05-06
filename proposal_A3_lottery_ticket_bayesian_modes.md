# 제안 A-3 — Lottery Ticket과 베이즈 사후 mode의 등가성

> **상위 트랙**: [A_mathematical_theoretical.md](A_mathematical_theoretical.md) §3.5, §6 제안 3
> **공격 대상 균열**: [99_synthesis.md](99_synthesis.md) §1.1 (균열 #1: 백프롭의 단일점 의존성) — 메타 어휘의 의문
> **층위 목표**: L2 (이론적) → L3 (개념적). "학습 = 최적화" 어휘를 "학습 = 추론"으로 재서술.
> **자원 등급**: 단일 RTX 5090 (32 GB), 4–6개월
> **목표 venue**: ICLR / TMLR / UAI

---

## 1. 동기 — 왜 이 질문이 지금 중요한가

Lottery Ticket Hypothesis (LTH)는 2019년 ICLR에서 발표된 이래 분야의 가장 매력적인 미스터리 중 하나로 남아 있다. Frankle & Carbin (2019)이 보인 핵심 사실 — **큰 신경망 안에는, 그 자체로 훈련해도 같은 정확도를 내는 작고 sparse한 부분망이 존재한다, 단 원래의 초기화를 유지해야 한다** — 은 두 종류의 후속 연구를 낳았다.

첫 번째는 **공학적**이다. lottery ticket을 효율적으로 찾는 알고리즘 (Iterative Magnitude Pruning, edge-popup, SNIP 등). 두 번째는 **존재론적**이다. Ramanujan et al. (2020) CVPR이 보인 충격적 결과 — **가중치를 전혀 학습하지 않고도 좋은 sub-network를 찾는 것만으로** SOTA에 근접 — 은 다음 질문을 강제한다:

> 학습은 정말 "정보를 더하는 것"인가, 아니면 "이미 있는 정보를 발굴하는 것"인가?

만약 후자라면, "학습 = 그래디언트 하강"이라는 1986년의 어휘 자체가 잘못이다. 학습은 **선택 (selection)** 이고, 선택의 자연스러운 수학적 언어는 **베이즈 추론**이다. 이는 [§99 균열 #1](99_synthesis.md) — 백프롭의 단일점 의존성 — 의 가장 깊은 메타 진단과 직결된다.

본 제안의 가설은 이 두 노선을 **하나의 수학적 진술**로 묶는다.

---

## 2. 배경 — 세 갈래의 결과들이 어떻게 만나는가

### 2.1 Lottery Ticket Hypothesis (선택)

- **Frankle & Carbin (2019)** ICLR. Iterative Magnitude Pruning (IMP)로 발견된 sparse subnetwork는 (a) 원래 초기화 보존 시 단독 학습 가능, (b) random 초기화로는 같은 정확도 도달 불가.
- **Frankle, Dziugaite, Roy, Carbin (2020)** ICML. _Linear Mode Connectivity and the Lottery Ticket Hypothesis_. 초기 K step 후 lottery ticket이 안정화되며, IMP는 손실 풍경 위 같은 mode basin에서 작동.
- **Ramanujan, Wortsman, Kembhavi, Farhadi, Rastegari (2020)** CVPR. _What's Hidden in a Randomly Weighted Neural Network?_ 무작위 가중치를 **전혀 훈련하지 않고도** mask만 선택하여 ImageNet에 근접. **학습 ≈ 선택**의 가장 강한 증거.

### 2.2 손실 풍경의 다중 mode 구조 (지형)

- **Garipov, Izmailov, Podoprikhin, Vetrov, Wilson (2018)** NeurIPS. _Loss Surfaces, Mode Connectivity, and Fast Ensembling_. 두 SGD 해는 손실이 거의 일정한 곡선으로 연결됨. mode는 점이 아니라 다양체.
- **Entezari, Sedghi, Saukh, Neyshabur (2022)** ICLR. _The Role of Permutation Invariance in Linear Mode Connectivity_. 순열 대칭성을 제거하면 두 mode는 사실상 같은 mode.
- **Draxler, Veschgini, Salmhofer, Hamprecht (2018)** ICML. 손실 장벽 없는 비선형 경로.

### 2.3 신경망의 베이즈 사후 (추론)

- **Welling & Teh (2011)** ICML. _Bayesian Learning via Stochastic Gradient Langevin Dynamics_. SGLD는 SGD에 적절한 노이즈를 더하면 사후 분포 샘플을 생성.
- **Mandt, Hoffman, Blei (2017)** JMLR. _Stochastic Gradient Descent as Approximate Bayesian Inference_. 정상 상태 SGD는 **온도 T = η/B**의 Gibbs 측도.
- **Wilson & Izmailov (2020)** NeurIPS. _Bayesian Deep Learning and a Probabilistic Perspective of Generalization_. 신경망 사후의 **다중 봉우리 (multimodal)** 구조가 ensemble의 본질.
- **Maddox, Izmailov, Garipov, Vetrov, Wilson (2019)** NeurIPS. _A Simple Baseline for Bayesian Uncertainty in Deep Learning_ (SWAG). SGD trajectory의 통계로 사후 근사.
- **Izmailov, Vikram, Hoffman, Wilson (2021)** ICML. _What Are Bayesian Neural Network Posteriors Really Like?_ HMC로 정확한 사후 샘플링. 진짜 봉우리 수는 작지만 mode 내 변동이 큼.

### 2.4 셋의 미접합 — 본 제안의 빈자리

세 갈래는 같은 대상 (학습된 신경망의 매개변수 공간) 을 다른 어휘로 묘사한다. 그러나 **셋을 동시에 다룬 작업은 없다**. 특히:

- LTH의 "winning ticket"이 사후의 **mode**에 해당하는지의 직접 검증이 없음.
- magnitude pruning이 **mode-finding의 휴리스틱**임을 명시한 논문이 없음 (암묵적 추측만 존재).
- 만약 등가성이 성립한다면, **사후를 직접 샘플링한 후 mode-seeking을 강제하는 pruning** 이 magnitude pruning을 대체할 후보가 된다.

이 빈자리가 본 제안의 표적이다.

---

## 3. 가설

### 3.1 주 가설 (H1)

> 학습된 신경망의 sparse subnetwork (lottery ticket)는 베이즈 사후 $p(\theta \mid \mathcal{D})$의 **mode**에 1:1 대응한다. magnitude pruning은 mode-finding의 휴리스틱이다.

조작적 정의:

- **Mode**: 사후의 local maximum의 basin. SGLD 샘플의 mean-shift clustering으로 식별.
- **Lottery ticket**: Frankle-Carbin IMP로 추출한 sparse subnetwork.
- **등가성**: 두 분포 (mode의 분포 vs ticket의 분포) 가 통계적으로 일치 — KS test, MMD, 2-Wasserstein 거리.

### 3.2 부속 가설

- **(H2)** 사후의 mode 수가 lottery ticket의 다양성을 결정한다. 작은 모델 (얕은 MLP) → mode 수 5–20개, 깊은 ResNet → 사후가 너무 많은 mode로 fragmented되어 IMP가 한 mode만 우선 선택.
- **(H3)** **변분 mode-finding pruning** (variational pruning) 은 magnitude pruning과 비슷한 정확도이나 **calibration이 본질적으로 우월**하다. mode를 명시적으로 추적하기 때문.
- **(H4)** Ramanujan et al.의 random-weight network는 **pre-trained 사후의 prior에서의 mode-finding**으로 해석된다. prior의 mode density가 SOTA 근접의 본질.

### 3.3 falsifiability

- mode 분포와 ticket 분포가 통계적으로 다르면 H1 기각.
- variational pruning이 magnitude pruning보다 본질적으로 열등하면 H3 기각.
- 두 경우 모두 "lottery ticket은 사후 구조와 무관한 별개 현상"의 negative evidence — 그 자체로 publication.

---

## 4. 방법

### 4.1 단계 1 — 사후 샘플링 (5090에 적합한 영역)

대상 모델 (작은 것부터 큰 것 순):

- **MLP-3** (3-layer FCN, 100K 파라미터) on **MNIST**, **Fashion-MNIST**
- **ResNet-20** (270K 파라미터) on **CIFAR-10**
- **VGG-11 small** (~2M 파라미터) on **CIFAR-100** (5090 한계 근처)

샘플러:

- **SGLD** (Welling-Teh 2011): 표준 SGD에 ${\cal N}(0, 2\eta T)$ 노이즈. preconditioned variant (Li et al. 2016) 으로 수렴 가속.
- **cyclical SGLD** (Zhang et al. 2020 ICLR): 주기적 high-temperature 단계로 mode-jumping. 다중 mode 발견에 본질적.
- 비교용: **HMC** (Izmailov et al. 2021의 reduced version) — 작은 모델만, ground truth.

샘플 수: 모델당 5,000–20,000 샘플 (chain 5개, chain당 1,000–4,000).

### 4.2 단계 2 — Mode 식별

연속 매개변수 공간에서 "mode"의 정의는 trivially 어렵다. 본 제안의 접근:

1. **순열 대칭성 제거** (Entezari 2022): activation matching으로 chain 간 align.
2. **저차원 투영**: PCA로 매개변수의 top 50차원 보존 (Maddox SWAG와 동형).
3. **mean-shift clustering** on 저차원 표상 → cluster center가 mode.
4. **basin entropy** (Bonatto et al. 2018, Phys. Rev. Lett.) 으로 mode 너비 측정.

### 4.3 단계 3 — Lottery Ticket 추출

Frankle-Carbin (2019) 표준 IMP:

- 90% 가중치를 magnitude로 prune.
- 원래 초기화로 rewind.
- 재훈련 → 다시 prune → 반복.

**Critical**: 동일 base 모델에서 **다중 random seed × 다중 IMP run** → ticket의 분포를 얻음 (단일 ticket이 아님).

### 4.4 단계 4 — 등가성 검정

ticket 분포 $\mathcal{T}$ vs mode 분포 $\mathcal{M}$ 비교:

- **층별 sparsity rate vector**의 KS test
- **mask Hamming 거리**의 분포 비교
- **활성화 공간**에서의 representation similarity (CKA, Kornblith et al. 2019)
- 2-Wasserstein 거리 ($\mathcal{T}, \mathcal{M}$이 모두 매개변수 공간 분포)

만약 등가성 성립 시:

- 매 ticket이 어느 mode에 매핑되는가? (Hungarian matching)
- mode 수와 ticket 다양성의 상관

### 4.5 단계 5 — 역방향: Variational Pruning

가설 H3의 직접 검증. 명시적 mode-seeking pruning 알고리즘:

```
def variational_prune(model, dataset, target_sparsity):
    # mask는 Bernoulli(p_ij)로 parameterize, p_ij는 학습 가능
    # Loss = E_{mask~q(mask;p)}[NLL(model⊙mask, dataset)]
    #        + λ * KL(q || prior_uniform)        # mode-seeking
    #        + μ * (sparsity - target)^2          # sparsity 강제
    optimize p via Adam
    return mask = round(p > 0.5)
```

비교 baseline: magnitude pruning, edge-popup (Ramanujan), SNIP, SynFlow.

평가:

- 정확도 (당연한 baseline)
- **calibration** (ECE, Brier score) — mode 추적의 본질적 이득이 여기서 나타나야
- mask 다양성 (다른 seed → 다른 mask?)

---

## 5. 5090 자원 매핑

### 5.1 메모리 예산

| 컴포넌트                                    | VRAM       |
| ------------------------------------------- | ---------- |
| MLP-3 + SGLD chain                          | <1 GB      |
| ResNet-20 + SGLD chain × 5                  | 4 GB       |
| VGG-11 small + cyclical SGLD                | 12 GB      |
| Variational pruning (gradient through mask) | +6 GB      |
| **Peak (worst case)**                       | **~20 GB** |

5090 32 GB에 충분히 fit.

### 5.2 Wall-clock

| Task                                       | 시간                 |
| ------------------------------------------ | -------------------- |
| MLP-3 SGLD 20K samples × 3 datasets        | 3일                  |
| ResNet-20 SGLD 5K samples × 5 chains       | 14일                 |
| VGG-11 cyclical SGLD 3K samples × 3 chains | 21일                 |
| IMP × 30 seed × 3 model                    | 14일                 |
| Variational pruning sweep                  | 21일                 |
| Mode clustering + analysis                 | 7일                  |
| **합계**                                   | **~3개월 핵심 실험** |

추가 buffer 2개월 (예상치 못한 디버깅 + ablation + 논문 작성).

### 5.3 소프트웨어 스택

- PyTorch 2.x + functorch (vmap으로 chain 병렬화)
- BoTorch / Pyro (HMC ground truth)
- SWAG official codebase (Maddox 2019 baseline)
- Frankle's open_lth (LTH baseline)
- 자체 구현: cyclical SGLD, variational pruning

---

## 6. 측정 — 무엇을 보면 가설이 맞다고 결론짓는가

### 6.1 H1 (mode ≡ ticket) 검증

| Metric                            | 결정 임계                     |
| --------------------------------- | ----------------------------- |
| 층별 sparsity KS p-value          | > 0.1 (분포 동일 reject 못함) |
| mask Hamming distribution overlap | > 70%                         |
| 활성화 CKA (mode vs ticket pair)  | > 0.85                        |
| Hungarian matching cost (정규화)  | < 0.3                         |

전부 만족 → H1 강한 지지. 일부만 → "부분 등가성" — 흥미로운 회색지대 결과.

### 6.2 H3 (variational vs magnitude) 검증

| Metric                     | 예상                                  |
| -------------------------- | ------------------------------------- |
| Test accuracy              | ± 1–2% (동등)                         |
| Expected Calibration Error | variational < magnitude (10–30% 개선) |
| Mask diversity (seed 간)   | variational > magnitude               |
| OOD detection AUC          | variational > magnitude               |

### 6.3 부정적 결과 시 contribution

- "Lottery ticket은 사후 mode와 통계적으로 다른 객체" → LTH의 메커니즘이 사후 구조 외의 무엇 (e.g., trajectory dependence) 임을 시사. 이는 Frankle 2020의 "stability" 결과와 충돌하므로 **재검증 자체가 중요**.

---

## 7. 위험 분석

| 위험                                                        | 확률 | 완화                                                               |
| ----------------------------------------------------------- | ---- | ------------------------------------------------------------------ |
| SGLD가 깊은 네트워크에서 mix 안 됨 (Izmailov 2021)          | 높음 | 작은 모델로 검증 후 deep extrapolation. cyclical SGLD + tempering. |
| 매개변수 공간 mode 정의의 자의성                            | 중   | activation 공간에서도 분석. CKA를 보조 metric으로.                 |
| Permutation symmetry로 mode 수가 폭발                       | 높음 | Entezari 2022 alignment 적용. **권장**: 항상 align 후 분석.        |
| Mask 비교가 noisy (sparsity rate가 낮으면 mask는 거의 same) | 중   | sparsity 50%, 70%, 90% 세 개 동시 ablation.                        |
| Variational pruning이 unstable                              | 중   | continuous relaxation (Gumbel-softmax) + temperature annealing.    |
| Compute가 wallclock 한계                                    | 중   | MLP-3과 ResNet-20에서 핵심 결과 도출, VGG-11은 stretch goal.       |

### Compute escape hatch

만약 ResNet-20 SGLD가 너무 느리면:

- chain 수 5 → 3
- chain당 sample 5K → 2K
- 결과: 통계적 power 감소, 그러나 trend는 보존.

---

## 8. 논문 구조 (목차 초안)

```
1. Introduction
2. Related Work
   2.1 Lottery Ticket Hypothesis
   2.2 Bayesian Deep Learning
   2.3 Mode Connectivity and Linear Mode Connectivity
3. Hypothesis: Tickets are Posterior Modes
4. Experimental Setup
   4.1 Posterior Sampling (SGLD, cyclical SGLD, HMC ground truth)
   4.2 Mode Identification
   4.3 Lottery Ticket Extraction
5. Equivalence Tests
   5.1 Mask Distribution Comparison
   5.2 Activation Space Alignment
   5.3 Hungarian Matching
6. Variational Pruning
   6.1 Algorithm
   6.2 Comparison to Magnitude Pruning
   6.3 Calibration and OOD Detection
7. Discussion
   7.1 Implications for the Selection View of Learning
   7.2 Connection to Ramanujan et al. (2020)
   7.3 Limitations: Deep Networks
8. Conclusion
```

---

## 9. 30개월 박사과정 학생의 첫 논문 적합도

본 제안은 다음 측면에서 **단독 박사과정 학생의 첫 NeurIPS/ICLR 논문**으로 거의 이상적이다:

- 코드베이스가 작음 (PyTorch + BoTorch + functorch)
- 실험이 결정적 (sweep + analysis, agent training 같은 "잘못 되면 다 재실행" 위험 낮음)
- 부정적 결과도 publishable
- 이론적 메시지가 깊음 (어휘 전환의 다리)
- 5090 단일 카드로 종결

이 점에서 같은 트랙의 A-2 (Grokking phase transition) 와 함께 본 연구실의 **Phase 1 워밍업** 후보로 가장 적합하다.

---

## 10. 어떤 균열을 공격하는가

| 균열 (`99_synthesis.md` §1)      | 본 제안의 공격 각도                                                                                                                                                                  |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **#1 백프롭의 단일점 의존성**    | "그래디언트 하강"이라는 **어휘 자체**의 의문. 학습이 사후 mode-finding이라면, 그 mode를 **다른 방법으로** 찾는 알고리즘 (variational, evolutionary, even forward-only) 이 모두 후보. |
| **#5 에피소드 메모리/즉시 학습** | mode 다양성이 학습된 표상의 ensemble을 형성. 이 ensemble이 즉시 학습의 prior로 작동 가능.                                                                                            |

§99 §4의 **Q2 (백프롭의 단일점 의존성을 어떻게 해소할 것인가)** 에 대한 직접적 부분 답: "사후 mode-finding으로 환원하면 알고리즘 디자인 공간이 폭발적으로 열린다."

---

## 11. 핵심 참고 (재정리)

### Lottery Ticket Hypothesis

- Frankle, J., & Carbin, M. (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. _ICLR_. arXiv:1803.03635.
- Frankle, J., Dziugaite, G. K., Roy, D. M., & Carbin, M. (2020). Linear Mode Connectivity and the Lottery Ticket Hypothesis. _ICML_.
- Ramanujan, V., Wortsman, M., Kembhavi, A., Farhadi, A., & Rastegari, M. (2020). What's Hidden in a Randomly Weighted Neural Network? _CVPR_.
- Lee, N., Ajanthan, T., & Torr, P. H. S. (2019). SNIP: Single-shot Network Pruning based on Connection Sensitivity. _ICLR_.
- Tanaka, H., Kunin, D., Yamins, D. L. K., & Ganguli, S. (2020). Pruning neural networks without any data by iteratively conserving synaptic flow (SynFlow). _NeurIPS_.

### Loss Landscape & Mode Connectivity

- Garipov, T., Izmailov, P., Podoprikhin, D., Vetrov, D., & Wilson, A. G. (2018). Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs. _NeurIPS_.
- Entezari, R., Sedghi, H., Saukh, O., & Neyshabur, B. (2022). The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks. _ICLR_.
- Draxler, F., Veschgini, K., Salmhofer, M., & Hamprecht, F. A. (2018). Essentially No Barriers in Neural Network Energy Landscape. _ICML_.

### Bayesian Deep Learning & SGLD

- Welling, M., & Teh, Y. W. (2011). Bayesian Learning via Stochastic Gradient Langevin Dynamics. _ICML_.
- Mandt, S., Hoffman, M. D., & Blei, D. M. (2017). Stochastic Gradient Descent as Approximate Bayesian Inference. _JMLR_.
- Zhang, R., Li, C., Zhang, J., Chen, C., & Wilson, A. G. (2020). Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning. _ICLR_.
- Maddox, W. J., Izmailov, P., Garipov, T., Vetrov, D., & Wilson, A. G. (2019). A Simple Baseline for Bayesian Uncertainty in Deep Learning (SWAG). _NeurIPS_.
- Wilson, A. G., & Izmailov, P. (2020). Bayesian Deep Learning and a Probabilistic Perspective of Generalization. _NeurIPS_.
- Izmailov, P., Vikram, S., Hoffman, M. D., & Wilson, A. G. (2021). What Are Bayesian Neural Network Posteriors Really Like? _ICML_.

### Variational Inference & Pruning

- Molchanov, D., Ashukha, A., & Vetrov, D. (2017). Variational Dropout Sparsifies Deep Neural Networks. _ICML_.
- Louizos, C., Welling, M., & Kingma, D. P. (2018). Learning Sparse Neural Networks through L₀ Regularization. _ICLR_.

### Permutation & Alignment

- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited (CKA). _ICML_.
- Ainsworth, S. K., Hayase, J., & Srinivasa, S. (2023). Git Re-Basin: Merging Models modulo Permutation Symmetries. _ICLR_.

---

> **요약**: 단일 RTX 5090로 4–6개월에 LTH의 **메타-수학적** 의문 — "lottery ticket은 무엇의 객체인가" — 에 대한 정량 답을 시도한다. 등가성이 성립하면 학습의 어휘가 "최적화 → 추론"으로 전환되는 첫 구체적 다리. 부정적 결과도 LTH의 메커니즘을 사후 구조 외부에서 찾도록 강제하는 정보. 어느 쪽이든 ICLR/TMLR-급 단일 학생 박사 주제 첫 논문으로 적합.
