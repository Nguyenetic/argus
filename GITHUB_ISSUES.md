# GitHub Issues to Create

Create these issues on https://github.com/Nguyenetic/argus/issues

---

## Issue #1: Implement Discrete SAC (Stable Discrete Soft Actor-Critic) RL Agent

**Labels:** `enhancement`, `week-3`, `rl-agent`, `high-priority`

### Description

Implement a Discrete SAC reinforcement learning agent for anti-bot evasion based on comprehensive research findings (see RL_RESEARCH_2025.md).

### Background

Research shows Discrete SAC is optimal for Argus because:
- **Best sample efficiency** (critical for web scraping where actions = HTTP requests)
- **Entropy-based exploration** (natural unpredictability prevents pattern detection)
- **Stochastic policy** (different behavior each session)
- **Recent stability improvements** (SDSAC paper, Nov 2024)

### Implementation Plan

**Week 3 Deliverables:**

#### 1. State Space (15 dimensions)
- [x] Design state representation (see `crates/argus-rl/src/state.rs`)
- [ ] Implement state normalization
- [ ] Add state observation from browser events
- [ ] Create state serialization for replay buffer

**Features:**
- Timing: `time_since_last_request`, `avg_interval`, `variance`
- Behavioral: `mouse_entropy`, `scroll_score`, `interaction_count`
- Page: `load_time`, `dynamic_ratio`, `complexity`
- Detection: `captcha_detected`, `rate_limit_hit`, `access_denied`, `challenge_score`
- Context: `requests_in_session`, `success_rate`

#### 2. Action Space (8 discrete actions)
- [ ] Define action enum
- [ ] Implement action executor (integrates with chromiumoxide)
- [ ] Add human behavior emulation (Perlin noise, Gaussian curves)

**Actions:**
1. Wait (short: 0.5-2s)
2. Wait (long: 2-10s)
3. Scroll (smooth, small)
4. Scroll (smooth, large)
5. Mouse movement (random Perlin noise)
6. Mouse movement + click (Gaussian curve)
7. Interact (hover + click)
8. Navigate (new page with delay)

#### 3. Reward Function
- [ ] Design reward calculator
- [ ] Implement reward shaping

**Rewards:**
- +10: Successful page scrape
- +5: No detection signals
- +2: Human-like behavior score > 0.8
- -5: CAPTCHA encountered
- -10: Rate limited
- -20: Access denied / blocked
- -1: Time penalty (per second, encourage efficiency)

#### 4. Neural Networks (tch-rs)
- [ ] Set up tch-rs dependency
- [ ] Implement Actor network (State → Action probabilities)
- [ ] Implement Critic networks (2x for double Q-learning)
- [ ] Implement temperature parameter (α) for entropy tuning
- [ ] Add network save/load functionality

**Architecture:**
```
Actor:  State(15) → FC(256) → FC(256) → Softmax(8)
Critic: State(15) + Action(8) → FC(256) → FC(256) → Q-value (scalar)
```

#### 5. Replay Buffer
- [ ] Implement prioritized experience replay
- [ ] Add ring buffer with capacity 100K
- [ ] Implement priority sampling based on TD-error
- [ ] Add buffer serialization for checkpointing

#### 6. Training Loop
- [ ] Implement SDSAC algorithm (arXiv:2209.10081)
- [ ] Add entropy-penalty (not bonus)
- [ ] Implement double average Q-learning with Q-clip
- [ ] Add gradient clipping
- [ ] Implement target network updates (soft updates with τ=0.005)

#### 7. Evaluation Environment
- [ ] Create synthetic bot detector (rule-based)
- [ ] Add ML-based detector (simple CNN)
- [ ] Implement adversarial co-evolution (detector retrains)
- [ ] Add evaluation metrics (evasion rate, detection rate)

### Success Criteria

- [ ] **>80% evasion rate** against synthetic detectors
- [ ] **<10K episodes** to converge
- [ ] **<100K transitions** sample efficiency
- [ ] **<24 hours** training time
- [ ] **Stable training** (no divergence)

### Technical Debt

- [ ] Unit tests for state/action/reward
- [ ] Integration tests with browser automation
- [ ] Benchmarks vs PPO and Rainbow DQN
- [ ] Documentation for RL agent usage

### References

- RL_RESEARCH_2025.md (comprehensive research)
- arXiv:2209.10081 (Stable Discrete SAC paper)
- https://github.com/coldsummerday/SD-SAC.git (PyTorch reference)
- https://github.com/LaurentMazare/tch-rs (Rust PyTorch bindings)

### Dependencies

- `tch-rs` for neural networks
- `argus-browser` for action execution
- `argus-core` for types and errors

---

## Issue #2: Add Human Behavior Emulation (Mouse Movement & Scrolling)

**Labels:** `enhancement`, `week-3-4`, `stealth`, `medium-priority`

### Description

Implement realistic human behavior emulation for mouse movements and scrolling patterns to avoid bot detection.

### Research Findings

From RL_RESEARCH_2025.md:
- **Gaussian Algorithm** - Gaussian random walks + Bezier curves
- **Perlin Noise** - Smooth randomness, avoids harsh transitions
- **DMTG Framework (Oct 2024)** - Diffusion model with entropy control

### Implementation

#### 1. Mouse Movement Emulation
- [ ] Implement Perlin noise algorithm for natural randomness
- [ ] Add Gaussian curve generation for realistic paths
- [ ] Implement variable speed (acceleration/deceleration)
- [ ] Add hover delay before clicks (100-500ms)
- [ ] Randomize click positions (not pixel-perfect center)

**Algorithm:**
```rust
pub struct MouseEmulator {
    perlin: PerlinNoise,
}

impl MouseEmulator {
    pub fn generate_path(start: Point, end: Point) -> Vec<Point> {
        // Use Bezier curves with Perlin noise
        // Variable speed with natural acceleration
    }
}
```

#### 2. Scrolling Patterns
- [ ] Implement variable scroll speeds
- [ ] Add natural deceleration at end
- [ ] Randomize scroll depth (10-90% of page)
- [ ] Add reading pauses (1-3s at content sections)
- [ ] Detect scroll "anchors" (headings, images)

#### 3. Timing Variations
- [ ] Log-normal distribution for inter-request delays
- [ ] Mean=5s, std=2s, ±30% variance
- [ ] Implement session patterns (burst → pause → burst)
- [ ] Add "human fatigue" (slower over time)

### Success Criteria

- [ ] **>0.8 human-like score** from behavioral analysis
- [ ] **No linear paths** detected
- [ ] **Natural timing** distribution
- [ ] **Passes CNN-based bot detection**

### Integration

- Integrate with RL agent action executor
- Called when action is "Mouse movement" or "Scroll"
- Configurable parameters (speed, randomness level)

---

## Issue #3: Add In-Memory Testing for NexusQL Storage

**Labels:** `enhancement`, `testing`, `quick-win`, `low-priority`

### Description

Add `:memory:` database support to Argus tests for 16.7× faster test execution with zero Windows file locking errors.

### Background

NexusQL v2.6.0 added in-memory database support (see NEXUSQL_UPDATE_v2.6.0.md).

### Implementation

```rust
#[cfg(test)]
impl NexusStorage {
    pub async fn new_memory() -> Result<Self> {
        Self::new(":memory:").await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_save_and_retrieve() {
        let storage = NexusStorage::new_memory().await.unwrap();
        // Test operations...
    }
}
```

### Benefits

- ✅ **16.7× faster** batch operations
- ✅ **Zero file cleanup errors** (Windows EPERM)
- ✅ **Perfect test isolation** (each `:memory:` is independent)
- ✅ **Industry standard** approach (matches SQLite/PostgreSQL)

### Effort

**Time:** 15-30 minutes
**Difficulty:** Easy
**Impact:** High (developer experience)

---

## Issue #4: Implement Vector Embeddings for Semantic Search

**Labels:** `enhancement`, `week-4-5`, `semantic-search`, `medium-priority`

### Description

Add vector embeddings to scraped pages for semantic search using NexusQL's HNSW index.

### Background

NexusQL already has:
- ✅ HNSW index created (`idx_pages_vector`)
- ✅ `content_vector VECTOR(384)` column
- ❌ No embeddings generated yet

### Implementation

#### 1. Choose Embedding Model
**Options:**
- **sentence-transformers** (all-MiniLM-L6-v2, 384 dims)
- **OpenAI** (text-embedding-3-small, 1536 dims)
- **Local model** (onnx-runtime for Rust)

**Recommendation:** Start with sentence-transformers via Python bridge, migrate to Rust later.

#### 2. Generate Embeddings
- [ ] Add embedding generation function
- [ ] Batch process existing scraped pages
- [ ] Generate embeddings on scrape (real-time)
- [ ] Handle long documents (chunking strategy)

#### 3. Update Storage Layer
```rust
pub async fn save_page_with_embedding(&self, page: &ScrapedPage, embedding: Vec<f32>) -> Result<()> {
    self.db.execute(
        "INSERT INTO pages (..., content_vector) VALUES (..., $8)",
        ExecuteParams::new(vec![..., embedding.into()])
    ).await?;
}

pub async fn vector_search(&self, query_embedding: Vec<f32>, limit: usize) -> Result<Vec<ScrapedPage>> {
    self.db.query(
        "SELECT * FROM pages ORDER BY content_vector <-> $1 LIMIT $2",
        ExecuteParams::new(vec![query_embedding.into(), (limit as i64).into()])
    ).await?;
}
```

#### 4. Add CLI Command
```bash
cargo run -- search "machine learning tutorials" --semantic --limit 10
```

### Success Criteria

- [ ] **Vector search API** implemented
- [ ] **Embeddings generated** for all pages
- [ ] **<100ms search** latency (p95)
- [ ] **>0.8 relevance** score for semantic queries

### Dependencies

- Embedding model (sentence-transformers or OpenAI)
- Python bridge or Rust ONNX runtime
- NexusQL v2.6.0+ (already integrated)

---

## Issue #5: Update TODO.md to Reflect Completed Work

**Labels:** `documentation`, `housekeeping`, `low-priority`

### Description

Update TODO.md to check off completed items and add new tasks from RL research.

### Completed Items

Week 1:
- [x] Set up Rust workspace
- [x] Create basic CLI tool
- [x] Implement simple HTTP scraping
- [x] Add JSON storage to `./data/`

Week 2:
- [x] Install and configure `chromiumoxide`
- [x] Create browser pool manager
- [x] Implement stealth techniques (7 techniques)
- [x] Add timeout and retry logic

CLI Enhancements:
- [x] Add progress bars for scraping operations
- [x] Implement parallel scraping with tokio tasks
- [x] Add export formats (CSV, Markdown, HTML)
- [x] Add filtering/search capabilities in `list` command
- [x] Implement `delete` command to remove scraped pages

### New Items to Add

Week 3: Basic RL Agent
- [ ] Implement Discrete SAC (not DQN as originally planned)
- [ ] Use tch-rs (not tch-rs mentioned in original TODO)
- [ ] Add human behavior emulation (Perlin + Gaussian)
- [ ] Create synthetic bot detection environment
- [ ] Target >80% evasion rate (same as original)

---

## Issue #6: Create Progress Tracking Dashboard

**Labels:** `enhancement`, `monitoring`, `low-priority`

### Description

Add a simple progress dashboard to track scraping statistics, RL training progress, and system metrics.

### Features

#### 1. Scraping Dashboard
- Total pages scraped
- Success rate (%)
- Average scrape time
- Detection rate (CAPTCHA, rate limits, blocks)
- Top domains scraped

#### 2. RL Training Dashboard
- Current episode
- Average reward (trailing 100 episodes)
- Evasion rate
- Exploration vs exploitation ratio
- Loss curves (actor, critic, temperature)

#### 3. System Metrics
- Browser pool utilization
- Memory usage
- Request rate (pages/sec)
- Cache hit rate (NexusQL)

### Implementation

**Option A:** TUI with ratatui
**Option B:** Web dashboard with Axum + htmx
**Option C:** Simple terminal output with formatted tables

**Recommendation:** Start with Option C, upgrade to Option A if needed.

---

## Priority Order

**Immediate (Week 3):**
1. **Issue #1** - Implement Discrete SAC RL Agent (HIGH PRIORITY)
2. **Issue #2** - Add Human Behavior Emulation (MEDIUM PRIORITY)

**Quick Wins:**
3. **Issue #3** - In-Memory Testing (15 min, high impact)
5. **Issue #5** - Update TODO.md (10 min)

**Future (Week 4-5):**
4. **Issue #4** - Vector Embeddings for Semantic Search
6. **Issue #6** - Progress Tracking Dashboard

---

## Notes

- All issues reference comprehensive research in RL_RESEARCH_2025.md
- Code backups on GitHub: https://github.com/Nguyenetic/argus
- Commit hash: a096e20 (Sessions 1-3 complete)
- All tests passing, zero errors, clean build
