# Deep RL Research for Anti-Bot Evasion (2025)

**Research Date:** November 4, 2025
**Purpose:** Determine optimal RL architecture for Argus web intelligence system

---

## Executive Summary

Based on comprehensive research of 2024-2025 literature, **Discrete SAC (Stable Discrete Soft Actor-Critic)** emerges as the optimal choice for Argus, with **PPO** as a strong fallback and **Rainbow DQN** for comparison benchmarking.

**Key Finding:** Modern bot evasion requires a combination of:
1. **Adaptive RL agent** (Discrete SAC/PPO) for behavioral decision-making
2. **Human behavior emulation** (Gaussian curves + Perlin noise for mouse movement)
3. **Entropy-controlled randomness** to avoid pattern detection

---

## Part 1: RL Algorithm Comparison (2024-2025 State-of-the-Art)

### 1.1 Algorithm Landscape

| Algorithm | Action Space | Sample Efficiency | Stability | Exploration | Best For |
|-----------|--------------|-------------------|-----------|-------------|----------|
| **DQN** | Discrete only | Low | Moderate | ε-greedy | Simple problems |
| **Double DQN** | Discrete only | Low-Moderate | Better | ε-greedy | Reduces overestimation |
| **Rainbow DQN** | Discrete only | High | High | Noisy Nets | Complex discrete tasks |
| **PPO** | Both | Moderate | Very High | Policy-based | General purpose, wall-clock critical |
| **SAC** | Continuous | Very High | High | Entropy-based | Continuous control |
| **Discrete SAC** | Discrete only | Very High | High (with SDSAC) | Entropy-based | Discrete + sample efficiency |
| **TD3** | Continuous only | Very High | Very High | N/A | Continuous deterministic |

### 1.2 Rainbow DQN (6 Improvements Combined)

**Components:**
1. **Double Q-learning** - Reduces overestimation bias
2. **Prioritized Experience Replay** - Samples important transitions more frequently
3. **Dueling Networks** - Separates state value and action advantage
4. **Multi-step Learning** - Captures longer-term dependencies
5. **Distributional RL** - Models entire return distribution
6. **Noisy Nets** - Parameter noise for better exploration

**Ablation Study Results (2024):**
- **Most Important:** Distributional head, multi-step learning, prioritized replay
- **Marginal Impact:** Dueling structure, double Q-learning (in context of Rainbow)
- **Performance:** Rainbow matches DQN after 7M frames, surpasses all baselines by 44M frames

**Verdict:** Rainbow is the **gold standard for discrete action spaces** but is complex to implement.

### 1.3 PPO (Proximal Policy Optimization)

**Strengths:**
- ✅ Supports discrete and continuous action spaces
- ✅ Very stable training (trust region optimization)
- ✅ Little hyperparameter tuning required
- ✅ Fast wall-clock training time
- ✅ Widely used in production (OpenAI, DeepMind)

**Weaknesses:**
- ❌ Lower sample efficiency than SAC
- ❌ Sensitive to hyperparameter changes
- ❌ Can require more samples to converge

**2024 Status:** Industry standard for general-purpose RL

### 1.4 Discrete SAC (Latest: Stable Discrete SAC - SDSAC, Nov 2024)

**Background:**
- SAC originally designed for continuous action spaces
- Adapted to discrete using Gumbel-Softmax or direct summation
- **Problem:** Q-value underestimation and instability in discrete settings

**SDSAC Improvements (Nov 2024):**
- **Entropy-penalty** instead of entropy bonus
- **Double average Q-learning** with Q-clip
- **Result:** Stable training with high sample efficiency

**Strengths:**
- ✅ **Best sample efficiency** for discrete actions
- ✅ Strong exploration via entropy maximization
- ✅ Stochastic policy (good for unpredictability)
- ✅ Recent research validates stability (SDSAC)

**Weaknesses:**
- ❌ More complex than PPO
- ❌ Requires careful implementation
- ❌ Less mature tooling than PPO

**2024-2025 Status:** Cutting-edge for discrete action spaces

### 1.5 Comparison: PPO vs Discrete SAC vs Rainbow DQN

**Sample Efficiency:**
1. **Discrete SAC** > Rainbow DQN > PPO

**Training Stability:**
1. **PPO** > SDSAC ≈ Rainbow DQN

**Implementation Complexity:**
1. **PPO** (simplest) < Discrete SAC < Rainbow DQN (most complex)

**Exploration Quality:**
1. **Discrete SAC** (entropy-based) > Rainbow (Noisy Nets) > PPO (policy stochasticity)

**Wall-Clock Time:**
1. **PPO** (fastest) < Discrete SAC < Rainbow DQN

---

## Part 2: Anti-Bot Evasion Research (2024-2025)

### 2.1 Academic Research

**"Web Bot Detection Evasion Using Deep Reinforcement Learning" (ACM 2022, cited in 2024-2025):**
- Bots can use RL to **evade behavior-based detection**
- Evaluated evasion against:
  - Pre-trained detection frameworks
  - Re-trained frameworks (adaptive defense)
  - Multi-round adversarial training
- **Key Insight:** RL agents can learn to mimic human patterns dynamically

**"RELEVAGAN" (2023):**
- Deep RL + GAN for botnet evasion
- DRL agent attacks discriminator during training
- **Accuracy:** 98.59% evasion rate
- **Technique:** Semantic-aware sample crafting

### 2.2 Industry Practices (2024-2025)

**OpenAI Computer-Using Agent (CUA):**
- **Architecture:** GPT-4o vision + RL training
- **Approach:** Perception → Reasoning → Action loop
- **Performance:**
  - 58.1% success on WebArena
  - 87% success on WebVoyager
  - 38.1% success on OSWorld
- **Key Technique:** Chain-of-thought reasoning + screenshot analysis

**Modern Anti-Bot Landscape (2024):**
- 37% of global web traffic is bots
- $4.9B market for anti-bot services
- AI-based detection uses ML to adjust detection parameters
- **Defense Techniques:**
  - Behavioral biometrics (mouse movement, timing)
  - Machine learning pattern recognition
  - CAPTCHA challenges
  - Rate limiting

### 2.3 Human Behavior Emulation (2024 State-of-the-Art)

**Mouse Movement Simulation:**

**Algorithms:**
1. **Gaussian Algorithm** - Gaussian random walks + Bezier curves
2. **Perlin Noise** - Smooth randomness, avoids harsh transitions
3. **DMTG Framework (Oct 2024)** - Diffusion model with entropy control
   - Generates unique trajectories each time
   - "Controllable randomness" mimics human unpredictability
   - Bypasses CAPTCHA detectors

**Scrolling Patterns:**
- Varying speeds and extents
- Pauses at different content sections
- Natural deceleration/acceleration

**Click Behavior:**
- Randomized click positions (not pixel-perfect center)
- Variable timing between actions
- Realistic hover duration before clicks

**Detection Evasion Metrics:**
- **CNN-based bot detection:** Superior performance vs traditional ML
- **Visual analysis:** Deep learning on mouse movement heatmaps

---

## Part 3: Rust ML Framework Comparison

### 3.1 Options for Argus

| Framework | Maturity | RL Support | PyTorch Compat | Performance | Complexity |
|-----------|----------|------------|----------------|-------------|------------|
| **tch-rs** | High | Examples (A2C) | Native bindings | Excellent | Moderate |
| **Burn** | Medium | None yet | Can import ONNX | Good | Low |
| **Candle** | Medium | None | Transformer focus | Excellent | Moderate |
| **burn-candle** | Low | None | Burn + Candle | Good | Low |

### 3.2 Recommendation: **tch-rs**

**Reasons:**
1. ✅ Mature PyTorch bindings (C++ API)
2. ✅ **Has RL examples** (A2C implementation for Atari)
3. ✅ Active community (2024 updates)
4. ✅ Can leverage PyTorch ecosystem
5. ✅ Production-ready performance

**Alternatives:**
- **Burn:** Good for general ML, but no RL support yet
- **Candle:** Excellent for transformers, but focused on NLP/CV

**Development Path:**
1. Start with tch-rs for RL implementation
2. Optionally export trained model to ONNX
3. Use Burn for inference if needed (ONNX import support)

---

## Part 4: Recommended Architecture for Argus

### 4.1 Primary Recommendation: **Discrete SAC (SDSAC)**

**Rationale:**
1. **Sample Efficiency** - Critical for web scraping (each action = real HTTP request)
2. **Entropy Maximization** - Natural exploration matches "unpredictability" requirement
3. **Stochastic Policy** - Prevents pattern detection (different behavior each session)
4. **Recent Advances** - SDSAC (Nov 2024) solves stability issues
5. **State-of-the-Art** - Best performance for discrete actions in 2024-2025

**Implementation:**
- Use **tch-rs** for neural networks
- Implement SDSAC based on paper (arXiv:2209.10081, updated Nov 2024)
- GitHub reference: https://github.com/coldsummerday/SD-SAC.git (PyTorch)

### 4.2 Fallback: **PPO**

**Rationale:**
1. **Proven Stability** - Industry standard
2. **Simpler Implementation** - Faster development
3. **Good Enough** - Still effective for bot evasion
4. **Fast Training** - Lower wall-clock time

**When to Use PPO:**
- If SDSAC implementation is too complex
- If training time is more critical than sample efficiency
- For rapid prototyping and testing

### 4.3 Benchmark: **Rainbow DQN**

**Rationale:**
1. **Academic Baseline** - Gold standard for discrete RL
2. **Performance Ceiling** - Shows what's possible
3. **Comparison Metric** - Validate SAC/PPO against Rainbow

**When to Implement:**
- After SAC/PPO is working
- For performance comparison
- If sample efficiency becomes critical bottleneck

---

## Part 5: Proposed Implementation Plan

### Phase 1: Foundation (Week 3) - **Discrete SAC**

**State Space (15 dimensions):**
```rust
- Timing features (3): time_since_last_request, avg_interval, variance
- Behavioral features (3): mouse_entropy, scroll_score, interaction_count
- Page features (3): load_time, dynamic_ratio, complexity
- Detection signals (4): captcha, rate_limit, access_denied, challenge_score
- Context (2): requests_in_session, success_rate
```

**Action Space (8 discrete actions):**
```rust
1. Wait (short: 0.5-2s)
2. Wait (long: 2-10s)
3. Scroll (smooth, small)
4. Scroll (smooth, large)
5. Mouse movement (random Perlin noise)
6. Mouse movement + click (Gaussian curve)
7. Interact (hover + click)
8. Navigate (new page with delay)
```

**Reward Function:**
```rust
+10: Successful page scrape
 +5: No detection signals
 +2: Human-like behavior score > 0.8
 -5: CAPTCHA encountered
-10: Rate limited
-20: Access denied / blocked
 -1: Time penalty (per second, encourage efficiency)
```

**Network Architecture:**
- **Actor Network:** State (15) → FC(256) → FC(256) → Softmax(8) [action probabilities]
- **Critic Networks (2):** State (15) + Action (8) → FC(256) → FC(256) → Q-value
- **Temperature (α):** Learnable parameter for entropy tuning

### Phase 2: Behavior Emulation (Week 3-4)

**Mouse Movement:**
- **Algorithm:** Perlin noise + Gaussian curves
- **Library:** Implement in Rust or call Python (pynput)
- **Features:** Speed variation, curved paths, hover delay

**Scrolling:**
- **Pattern:** Randomized scroll depth (10-90% of page)
- **Speed:** Variable (fast skim vs slow read)
- **Pauses:** Natural reading pauses (1-3s)

**Timing:**
- **Inter-request delay:** Log-normal distribution (mean=5s, std=2s)
- **Randomization:** ±30% variance per action
- **Session patterns:** Burst → pause → burst (human-like sessions)

### Phase 3: Training Environment (Week 4)

**Synthetic Bot Detection:**
- **Rule-based detector:** Flags low entropy, high frequency, straight mouse paths
- **ML-based detector:** Simple CNN trained on synthetic data
- **Adaptive detector:** Re-trains after N episodes (adversarial co-evolution)

**Training Process:**
1. Initialize SDSAC networks
2. Generate episodes (100 steps each)
3. Collect transitions: (state, action, reward, next_state, done)
4. Store in replay buffer (size: 100K)
5. Sample batches (size: 256) and update networks
6. Evaluate every 10 episodes on hold-out detector
7. Target: >80% evasion rate after 10K episodes

### Phase 4: Integration (Week 5)

**Browser Automation + RL:**
```rust
1. RL agent selects action (e.g., "Scroll smooth, small")
2. Browser pool executes action via chromiumoxide
3. Observe next state (page features, detection signals)
4. Calculate reward based on outcome
5. Store transition in replay buffer
6. Update RL networks every N steps
```

**Inference Mode:**
- Load trained SDSAC model
- Use **actor network only** (no critic needed)
- Select actions stochastically from policy
- Apply human behavior emulation (Perlin mouse, Gaussian scroll)

---

## Part 6: Performance Targets

### Training Metrics (Week 3-4)

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| Evasion Rate | >80% | >90% |
| Episodes to Converge | <10K | <5K |
| Training Time | <24 hours | <12 hours |
| Sample Efficiency | <100K transitions | <50K transitions |

### Production Metrics (Week 5+)

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| Scraping Success Rate | >90% | >95% |
| CAPTCHA Encounter Rate | <5% | <2% |
| Detection/Block Rate | <10% | <5% |
| Human-Like Score | >0.8 | >0.9 |

---

## Part 7: Risk Mitigation

### Technical Risks

**Risk 1: SDSAC Implementation Complexity**
- **Mitigation:** Start with PPO, migrate to SDSAC later
- **Fallback:** Use PPO as production algorithm

**Risk 2: tch-rs Learning Curve**
- **Mitigation:** Reference A2C example, PyTorch documentation
- **Fallback:** Implement in Python first, port to Rust

**Risk 3: Training Instability**
- **Mitigation:** Use SDSAC improvements (Q-clip, double averaging)
- **Fallback:** Rainbow DQN or PPO

### Ethical Risks

**Risk 1: Evasion Technology Misuse**
- **Mitigation:** Document authorized use cases (pentesting, research)
- **Policy:** Refuse destructive techniques, mass scraping

**Risk 2: Arms Race with Detection**
- **Mitigation:** Focus on "polite scraping" (rate limits, robots.txt)
- **Philosophy:** Evasion as defense against false positives, not malicious

---

## Part 8: Key Takeaways

### Algorithm Selection

**Winner: Discrete SAC (SDSAC)**
- Best sample efficiency (critical for web scraping)
- Entropy-based exploration (unpredictability)
- Stochastic policy (anti-pattern detection)
- Recent stability improvements (Nov 2024)

**Runner-Up: PPO**
- Simpler, faster, proven
- Good for rapid prototyping
- Lower risk implementation

**Benchmark: Rainbow DQN**
- Academic gold standard
- Performance ceiling
- Complex but powerful

### Implementation Strategy

**Week 3:** Implement Discrete SAC with tch-rs
**Week 4:** Add human behavior emulation (Perlin, Gaussian)
**Week 5:** Integrate with browser automation
**Week 6:** Train, benchmark, optimize

### Success Criteria

✅ **>80% evasion rate** against synthetic detectors
✅ **<10% block rate** in production
✅ **Human-like behavior score >0.8**
✅ **Sample efficient training** (<100K transitions)

---

## References

1. **Stable Discrete SAC (SDSAC)** - arXiv:2209.10081, updated Nov 2024
2. **Rainbow DQN** - Hessel et al., 2017, AAAI 2018
3. **Web Bot Detection Evasion Using DRL** - ACM ARES 2022
4. **RELEVAGAN** - Deep RL + GAN for Botnet Evasion, 2023
5. **OpenAI CUA** - Computer-Using Agent, 2024
6. **DMTG Framework** - Human-Like Mouse Trajectories, Oct 2024
7. **CleanRL** - Single-file RL implementations (PPO, SAC, DQN)
8. **tch-rs** - PyTorch Rust bindings, GitHub: LaurentMazare/tch-rs
9. **Stable Baselines3** - RL Tips and Tricks, 2024 documentation

---

## Appendix: Code References

### SDSAC Implementation (PyTorch)
- https://github.com/coldsummerday/SD-SAC.git

### CleanRL (Reference Implementations)
- https://github.com/vwxyzjn/cleanrl

### tch-rs A2C Example
- https://github.com/LaurentMazare/tch-rs/blob/main/examples/reinforcement-learning/a2c.rs

### Discrete SAC (PyTorch)
- https://github.com/alirezakazemipour/Discrete-SAC-PyTorch

---

**Conclusion:** Argus should implement **Discrete SAC (SDSAC)** as the primary RL algorithm, with **PPO** as fallback and **Rainbow DQN** for benchmarking. This architecture maximizes sample efficiency, exploration quality, and unpredictability—essential for anti-bot evasion in 2025.
