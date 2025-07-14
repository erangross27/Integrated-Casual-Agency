# ICA Framework Intelligence Testing Suite

This folder contains tools to objectively assess the intelligence capabilities of the ICA Framework AGI system.

## 🧪 Testing Tools

### 1. Knowledge Graph Analysis (`analyze_intelligence.py`)

**Purpose**: Quantifies the scale and sophistication of learned knowledge

**Usage**:
```bash
python analyze_intelligence.py
```

**What it measures**:
- **Knowledge Scale**: Total entities and relationships in the knowledge graph
- **Reasoning Depth**: Multi-hop reasoning chains (1-5 relationship hops)
- **Knowledge Confidence**: High-confidence vs exploratory learning patterns
- **Network Topology**: Hub entities and knowledge interconnectedness

**Sample Output**:
```
🧠 ICA Framework Intelligence Analysis
📊 Knowledge Scale: 93,449 entities | 205,891 relationships  
🔍 Multi-hop reasoning chains: 45,219 paths discovered
🎯 High-confidence knowledge: 31,204 entities (33.4%)
🔗 Hub entities: 156 concepts with 50+ connections
📈 Average reasoning depth: 2.8 relationship hops
```

**Interpretation**:
- **< 10K entities**: Early learning phase
- **10K-50K entities**: Basic intelligence emerging
- **50K-100K entities**: Competent knowledge acquisition
- **> 100K entities**: Advanced intelligence threshold
- **> 3.0 avg hops**: Strong multi-step reasoning capability

---

### 2. Real-World Intelligence Test (`test_real_intelligence.py`)

**Purpose**: Tests practical problem-solving abilities across 5 key domains

**Usage**:
```bash
python test_real_intelligence.py
```

**Test Domains**:

#### 🔋 Energy Efficiency (100 points)
- Tests pattern recognition in power consumption data
- Evaluates understanding of energy optimization strategies
- Measures ability to identify efficiency patterns

#### 🔗 Causal Reasoning (100 points)  
- Tests multi-step cause-effect understanding
- Evaluates logical reasoning chains
- Measures ability to trace complex causality

#### ⚙️ Optimization (100 points)
- Tests resource allocation capabilities
- Evaluates constraint satisfaction problem solving
- Measures strategic planning abilities

#### 🛡️ Safety Analysis (100 points)
- Tests risk assessment capabilities
- Evaluates hazard identification skills
- Measures safety protocol understanding

#### 🔮 Predictive Intelligence (100 points)
- Tests behavior forecasting abilities
- Evaluates trend analysis capabilities
- Measures future state prediction accuracy

**Scoring**:
- **0-100**: Minimal intelligence, basic pattern matching only
- **100-200**: Emerging intelligence in specific domains
- **200-300**: Competent intelligence with clear strengths
- **300-400**: Advanced intelligence across multiple areas  
- **400-500**: Expert-level AGI (exceptional achievement)

**Sample Output**:
```
🧠 Real-World Intelligence Assessment
🏆 Overall Score: 200/500 (40% - Emerging Intelligence)

📊 Domain Breakdown:
⚡ Energy Efficiency: 100/100 (Expert level)
🔗 Causal Reasoning: 100/100 (Strong logic)
⚙️ Optimization: 0/100 (Limited strategies)
🛡️ Safety Analysis: 0/100 (Basic awareness)
🔮 Predictive Intelligence: 0/100 (Minimal capability)
```

---

## 🚀 Best Practices for Testing

### 1. Run After Learning
For meaningful results, test after the AGI has learned for several hours:

```bash
# Start continuous learning
python run_continuous.py

# Let it run for 2+ hours, then stop with Ctrl+C

# Run intelligence tests
python intelligence_tests/analyze_intelligence.py
python intelligence_tests/test_real_intelligence.py
```

### 2. Progressive Testing
Test periodically to track intelligence growth:

```bash
# Initial baseline (after 30 minutes)
python intelligence_tests/analyze_intelligence.py

# Growth check (after 2 hours)  
python intelligence_tests/analyze_intelligence.py

# Maturity test (after 8+ hours)
python intelligence_tests/test_real_intelligence.py
```

### 3. Database Requirements
Both tools require:
- **Neo4j database** running and configured
- **Existing knowledge graph** from ICA Framework learning
- **Proper database connection** (configured via `python setup.py database`)

---

## 📊 Understanding Intelligence Growth

### Knowledge Scale Progression
```
🌱 Early (< 1 hour):     1K-10K entities
🌿 Developing (1-3 hours): 10K-50K entities  
🌳 Mature (3-8 hours):    50K-100K entities
🏔️ Advanced (8+ hours):   100K+ entities
```

### Reasoning Capability Milestones
```
🧠 Basic (< 2.0 hops):     Simple associations
🧠 Intermediate (2.0-3.0): Multi-step reasoning
🧠 Advanced (3.0-4.0):     Complex logical chains
🧠 Expert (4.0+ hops):     Deep analytical thinking
```

### Real-World Performance Benchmarks
```
🎯 Emerging (100-200/500):    Domain-specific strengths
🎯 Competent (200-300/500):   Multi-domain capabilities
🎯 Advanced (300-400/500):    Broad intelligence spectrum
🎯 Expert (400-500/500):      Human-level problem solving
```

---

## 🔬 Technical Details

### Dependencies
Both tools require the same dependencies as the main ICA Framework:
- Python 3.8+
- Neo4j database
- ICA Framework installed (`pip install -e .`)

### Data Source
Tests analyze the Neo4j knowledge graph created by:
- `run_continuous.py` (recommended for comprehensive testing)
- `examples/learning.py` (basic testing possible)

### Honest Assessment Philosophy
These tools provide **realistic intelligence assessment** rather than inflated metrics. The scoring is designed to:
- Reward genuine capabilities
- Penalize superficial pattern matching
- Provide actionable feedback for improvement
- Set realistic expectations for AGI development

---

## 🤝 Contributing

Found the AGI performing better than expected? Submit your test results:
1. Run both intelligence tests
2. Include learning duration and scenario count
3. Share knowledge graph metrics
4. Document any unexpected capabilities

**Help us improve AGI assessment!**
