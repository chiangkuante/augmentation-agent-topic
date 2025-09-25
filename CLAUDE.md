# CLAUDE.md

This file provides guidance to Claude Code when working with the "augmentation-agent-topic" repository.

## Project Status

This is a research project to implement an LLM-assisted iterative topic modeling framework. The initial data collection and preprocessing phase is complete. The primary data source is a CSV file named `corpus_semantic_chunks.csv`, located in the `/data` directory. The project's goal is to analyze corporate reports to derive a "digital resilience" score.

The project will be built by implementing a series of Python modules located in the `src/` directory.

## Setup Instructions

### Environment Setup

#### Quick Setup (Recommended)
Run the automated setup script:
```bash
python setup_environment.py
```

This script will:
- Check Python version and dependencies
- Create appropriate `.env` file for your environment
- Set up OpenAI API key
- Create necessary directories
- Validate configuration

#### Manual Setup
1. **Create conda environment:**
   ```bash
   mamba create -n augmentation-agent-topic python=3.10
   mamba activate augmentation-agent-topic
   ```

2. **Install dependencies:**
   ```bash
   mamba install --file requirements.txt
   # or
   pip install -r requirements.txt
   ```

3. **Environment Configuration:**
   ```bash
   # Copy appropriate environment template
   cp .env.development .env  # For development
   cp .env.production .env   # For production
   cp .env.example .env      # For custom setup
   ```

4. **Configure API Key:**
   Edit `.env` file and set:
   ```bash
   OPENAI_API_KEY=your-actual-api-key-here
   ```

5. **Initialize git repository:**
   ```bash
   git init
   git add .
   git commit -m "Initial setup with environment configuration"
   ```

#### Environment Files
- **`.env.example`**: Complete template with all options
- **`.env.development`**: Development defaults (low budget, small iterations)
- **`.env.production`**: Production defaults (higher budget, full features)
- **`.env.local`**: Local overrides (gitignored)
- **`.env`**: Your active configuration (gitignored)

### System Requirements
- **Memory:** Minimum 8GB RAM (16GB+ recommended for large corpora)
- **GPU:** Optional but recommended for ModernBERT inference
- **Storage:** 5-10GB for models and intermediate results
- **API Budget:** $50-100 recommended per pipeline run

### Common Commands

#### Configuration Management
- **Validate configuration:** `python src/main.py --validate-config`
- **Show current configuration:** `python src/main.py --show-config`
- **Switch environment:** `python src/main.py --env production`

#### Pipeline Execution
- **Run full pipeline:** `python src/main.py`
- **Development test run:** `python src/main.py --env development --sample-size 100`
- **Production run:** `python src/main.py --env production --budget 100.0`
- **Custom configuration:** `python src/main.py --model gpt-3.5-turbo --budget 25.0 --max-iterations 3`

#### Maintenance
- **Install additional packages:** `mamba install <package_name>`
- **Run tests:** `pytest tests/`
- **View logs:** Check `logs/` directory for execution logs
- **Environment setup:** `python setup_environment.py`

### Architecture Overview
- **Project Structure:**

.
├── .gitignore
├── config/
│   ├── prompts.py         # Centralized prompt templates
│   └── settings.py        # Project configuration
├── data/
│   └── corpus_semantic_chunks.csv
├── logs/                  # Execution logs
├── notebooks/            # Jupyter notebooks for experimentation
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── initial_topic_modeler.py
│   ├── llm_agent.py
│   ├── iterative_optimizer.py
│   ├── resilience_mapper.py
│   └── main.py
├── results/
│   ├── topics/
│   └── scores/
├── tests/               # Unit and integration tests
├── requirements.txt
└── environment.yml

- **Main Technologies:** Python 3.10+, `pandas`, `bertopic`, `openai`, `langchain`, `nltk`, `scikit-learn`, `tiktoken`, `plotly`, `transformers` (ModernBERT).
- **Package Manager:** `mamba` (recommended) or `conda`
- **Key Directories:**
- `src/`: All main application logic. Each file represents a module in the pipeline.
- `data/`: Input data.
- `results/`: All generated outputs, including topic lists and final scores.

## Development Plan & Key Modules

Please implement the following modules in order.

### 1. `src/data_loader.py`
- **Purpose:** To load and prepare the input data.
- **Key Class/Function:** `load_corpus(file_path)`
- **Input:** Path to `corpus_semantic_chunks.csv`.
- **Output:** A pandas DataFrame with columns `['ticker', 'year', 'text']`.
- **Implementation Notes:** Ensure correct handling of CSV encoding.

### 2. `src/initial_topic_modeler.py`
- **Purpose:** To perform the initial topic modeling using BERTopic.
- **Key Class/Function:** `create_initial_topics(documents)`
- **Input:** A list of texts (documents) from the data loader.
- **Output:** A trained `bertopic.BERTopic` model object and a DataFrame of the initial topics and their keywords.
- **Implementation Notes:**
- Use `BERTopic` from the `bertopic` library.
- **Embedding Model Options (Phased Approach):**
  - **Phase 1 (MVP):** `sentence-transformers/all-mpnet-base-v2` (proven stability)
  - **Phase 2 (Advanced):** `ModernBERT` with custom SentenceTransformer wrapper
  - **Phase 3 (Experimental):** Direct `transformers` library integration
- Use UMAP for dimensionality reduction and HDBSCAN for clustering, as is standard in BERTopic.
- Implement memory-efficient embedding computation with batching for large corpora.
- Save the initial model and topic list to the `/results/topics` directory.
- Add checkpointing system to recover from failures during initial modeling.

### 3. `src/llm_agent.py`
- **Purpose:** This module contains the core intelligence of the framework. It uses an LLM to analyze and suggest optimizations for the topic model.
- **Key Class/Function:** `LLMAgent` class.
- `__init__(self, api_key, model_name="gpt-4", temperature=0.1, max_retries=3)`:
  - Initializes the OpenAI client with configurable model selection
  - Supports models: `gpt-4`, `gpt-3.5-turbo`, `gpt-4-turbo`
  - Temperature parameter for output stability (default: 0.1)
  - Built-in retry mechanism for API failures
- `get_topic_name_and_summary(self, keywords)`: Takes a list of keywords and returns a JSON object `{"name": "...", "summary": "..."}`.
- `evaluate_topic_quality(self, topic_name, keywords)`: Returns a JSON `{"coherence": int, "uniqueness": int, "reason": "..."}`.
- `calculate_tokens_and_cost(self, text)`: Estimates token usage and API costs for budget control.
- `generate_optimization_commands(self, topics_data)`: This is the most critical function.
  - **Input:** A list of dictionaries, where each dictionary represents a topic (e.g., `{'id': 1, 'name': '...', 'keywords': [...], 'quality_scores': {...}}`).
  - **Logic:** Use a Chain-of-Thought (CoT) prompt to make the LLM reason about which topics to merge or split.
  - **CoT Prompt Template:**
    ```
    You are an expert in topic model optimization. Your task is to analyze the following list of topics and suggest actions to improve their quality.

    Here is the current list of topics:
    {topic_list_string}

    Please follow these steps in your reasoning:
    1.  **Identify Overlaps:** Are there any topics that are semantically very similar or subsets of each other? List them and explain why.
    2.  **Identify Poor Quality Topics:** Are there any topics with low quality scores or keywords that seem incoherent and generic? List them.
    3.  **Propose Actions:** Based on your analysis, propose a list of actions. The only valid actions are 'MERGE' and 'SPLIT'. 'MERGE' should target two or more topic IDs. 'SPLIT' should target a single topic ID.

    Your final output must be a JSON array of action objects. For example:
    [
      {"action": "MERGE", "targets": [3, 15], "reason": "Topics 3 and 15 both discuss cybersecurity risks and can be combined."},
      {"action": "SPLIT", "targets": [8], "reason": "Topic 8 is too broad, covering both hardware and software infrastructure, and should be separated."}
    ]
    ```
  - **Output:** A list of parsed action dictionaries.
- **Implementation Notes:**
  - Include comprehensive error handling with exponential backoff for API rate limits
  - Implement token counting to prevent exceeding context limits
  - Add logging for all API calls and responses
  - Use structured prompts stored in `config/prompts.py`
  - Include cost estimation and budget alerts

### 4. `src/iterative_optimizer.py`
- **Purpose:** To manage the iterative refinement loop with robust quality control and cost management.
- **Key Class/Function:** `Optimizer` class.
- `__init__(self, max_iterations=5, convergence_threshold=0.02, quality_history_window=2, api_budget_limit=50)`:
  - **Reduced maximum iterations to 5** (cost and time optimization)
  - Quality convergence threshold for early stopping
  - Smaller history window for faster plateau detection
  - **API budget limit with automatic stopping**
- `run(self, documents)`: The main loop with enhanced safeguards.
- **Phased Implementation Logic:**

  **Phase 1 (MVP) - MERGE-only operations:**
  1. Call `create_initial_topics` and record baseline quality metrics.
  2. Initialize cost tracking and budget monitoring.
  3. Loop for `max_iterations` with strict cost control:
     a. **Check API budget before each iteration** - stop if exceeded.
     b. Get current topics from the BERTopic model.
     c. Calculate quality metrics (coherence + silhouette + diversity).
     d. **Batch similar topics for analysis** to reduce API calls by 60-80%.
     e. Pass topic batches to `LLMAgent` to get MERGE commands only.
     f. If no commands or budget exceeded, break the loop.
     g. Execute MERGE commands using `bertopic_model.merge_topics()`.
     h. Re-calculate quality metrics and compare with previous iteration.
     i. **Multi-criteria convergence check:**
        - Quality improvement below threshold
        - Cost-benefit analysis (improvement vs. API cost)
        - Quality plateau detection
     j. **Automatic rollback** if quality degrades significantly.

  **Phase 2 (Advanced) - Add SPLIT operations:**
  - Implement hierarchical topic modeling for splitting
  - Use separate BERTopic instances for complex operations
  - Maintain document assignment consistency

- **Enhanced Quality Control & Cost Management:**
  - **Real-time API cost tracking** with hard budget limits
  - **Combined quality metrics:** LLM assessment + mathematical scores
  - **Rollback mechanism** to best previous model state
  - **Early stopping** based on cost-benefit analysis
  - **Checkpointing** after each successful iteration
  - **Detailed cost and quality reporting** for each run

### 5. `src/resilience_mapper.py`
- **Purpose:** To map the final topics to the comprehensive digital resilience framework and calculate detailed scores.

#### Digital Resilience Framework Definition

**1. Absorption Capacity (Absorb)**
Clear Definition: Absorption capacity refers to an organization's ability to withstand major impacts, maintain its core structure and operational methods, and ensure organizational survival. Its focus is on "defense" and "stability," with the goal of immediately reducing initial losses and ensuring uninterrupted operations when impacts occur.

Specific Standards:
- **Backup and Alternative Solutions**: Backup systems, alternative suppliers, or substitute operational models that can be quickly switched to when impacts occur. Examples: off-site backup data centers, diversified supply chain strategies to avoid single points of failure.
- **Prediction and Resistance Capabilities**: Using digital technologies (such as data analysis, AI) to collect and analyze data to predict and resist potential impacts. Examples: using big data to predict supply chain risks, employing AI for cybersecurity threat detection.
- **Operational Stability Maintenance**: Using digital technologies to maintain organizational operational stability during impacts. Examples: rapidly implementing remote work systems and online collaboration platforms during the pandemic to maintain daily operations.

**2. Adaptive Capacity (Adapt)**
Clear Definition: Adaptive capacity refers to an organization's ability to proactively adjust its structure, operational methods, and business strategies when facing impacts, to respond to new environments and conditions while maintaining operational flexibility and adaptability. Its focus is on "adjustment" and "flexibility," with the goal of making changes in response to environmental changes.

Specific Standards:
- **Process and Resource Reallocation**: Changing existing business processes, reallocating resources, or adjusting organizational structure in response to impacts. Examples: transferring physical channel resources to online channels in response to market changes and transforming related departmental functions.
- **Digital Response Solutions**: Using digital technologies to quickly respond to impacts by developing new digital solutions or service models. Examples: restaurants developing online ordering and delivery apps during the pandemic; manufacturers implementing digital twins to simulate and optimize production line adjustments.
- **Market and Product Repositioning**: Using digital channels or data analysis to reposition products and services to meet new market demands. Examples: using customer data analysis to adjust product features and engaging in new market communication through social media.

**3. Transformation Capacity (Transform)**
Clear Definition: Transformation capacity refers to an organization's ability to undergo deep-level, fundamental, or even revolutionary changes after experiencing major impacts, thereby transitioning to a new, more stable state to adapt to entirely new environments and conditions. Its focus is on "change" and "evolution," with the goal not only of responding to current impacts but also of establishing a new model that can adapt to future changes.

Specific Standards:
- **Fundamental Business Model Redesign**: Completely transforming core business models in response to impacts. Examples: traditional retailers transforming into online subscription service platforms; manufacturers shifting from selling products to providing "Product-as-a-Service" (PaaS).
- **Adoption and Integration of Emerging Digital Technologies**: Adopting new digital technologies (such as AI, blockchain) to achieve deep-level organizational transformation. Examples: comprehensively upgrading to smart factories using IoT and AI, fundamentally changing production and management models.
- **Ecosystem and Partnership Reconstruction**: Establishing new partnerships or ecosystems to create new value and stability. Examples: forming new industry alliances with startup tech companies or even competitors to jointly develop new technology platforms.

#### Key Functions:
- `map_topics_to_resilience(final_topics, framework_definition)`:
  - **Input:** The final list of topics and the comprehensive resilience framework definition above.
  - **Logic:** Use an LLM with detailed prompts to classify each topic into one or more dimensions based on the specific criteria above.
  - **Output:** A dictionary mapping each topic ID to its resilience dimension(s) with confidence scores.
- `calculate_resilience_scores(documents_with_topics, topic_resilience_map)`:
  - **Input:** A DataFrame containing all documents and their final topic assignments, and the mapping from the previous function.
  - **Logic:** For each company-year, calculate weighted scores based on document frequency and topic relevance to each resilience dimension.
  - **Output:** A DataFrame with `['ticker', 'year', 'absorb_score', 'adapt_score', 'transform_score', 'total_resilience_score', 'confidence_scores']`.

### 6. `src/main.py`
- **Purpose:** The main entry point to run the entire pipeline.
- **Logic:**
1.  Call `data_loader.load_corpus`.
2.  Instantiate and run the `Optimizer`.
3.  Get the final topics and document assignments.
4.  Call `resilience_mapper` functions.
5.  Save the final scores and topic lists to the `/results` directory.

## Implementation Phases & Validation

### Phase 1: MVP (Minimum Viable Product)
**Timeline:** 2-3 weeks
**Goals:** Prove core concept with cost control
- Use `sentence-transformers/all-mpnet-base-v2` for stable embeddings
- Implement MERGE-only topic operations
- Hard API budget limits ($50 per run)
- Maximum 5 iterations with early stopping
- Basic quality metrics (coherence + silhouette)

### Phase 2: Enhanced System
**Timeline:** 3-4 weeks additional
**Goals:** Add advanced features and robustness
- ModernBERT integration with custom wrapper
- SPLIT functionality via hierarchical modeling
- Local LLM support (Llama/Mistral) for cost reduction
- Advanced quality assessment combining multiple metrics
- Human evaluation validation samples

### Phase 3: Production Ready
**Timeline:** 2-3 weeks additional
**Goals:** Scalability and enterprise features
- Distributed processing for large corpora
- Real-time monitoring and alerting
- Advanced error recovery and checkpointing
- Comprehensive validation framework

### Validation Framework
**Quality Assurance Methods:**
1. **Mathematical Validation:** Coherence scores, silhouette analysis, topic diversity metrics
2. **Human Evaluation:** Sample-based topic quality assessment by domain experts
3. **Cross-Validation:** K-fold validation on document-topic assignments
4. **Benchmark Comparison:** Compare against baseline BERTopic without LLM optimization
5. **Resilience Score Validation:** Test cases with known resilience examples from literature

**Success Criteria:**
- Topic coherence improvement: >10% vs baseline
- API cost per run: <$50 for typical corpus sizes
- Processing time: <4 hours for 10K documents
- Human evaluator agreement: >0.7 kappa score on topic quality

## Risk Mitigation & Alternative Approaches

### Critical Risk Areas & Mitigation Strategies

**1. API Cost Explosion (HIGH RISK)**
- **Risk:** Uncontrolled API costs exceeding $500+ per run
- **Mitigation:**
  - Hard budget limits with automatic stopping
  - Local LLM fallback (Ollama + Llama-3.1-8B)
  - Batch processing to reduce API calls by 70%
  - Cost monitoring dashboard

**2. ModernBERT Integration Complexity (HIGH RISK)**
- **Risk:** Custom integration failures and embedding inconsistencies
- **Mitigation:**
  - Start with proven `sentence-transformers` models
  - Phased approach with fallback options
  - Comprehensive testing of embedding consistency

**3. Topic Splitting Implementation (MEDIUM-HIGH RISK)**
- **Risk:** Complex custom logic causing model corruption
- **Mitigation:**
  - MVP focuses on MERGE-only operations
  - Use hierarchical BERTopic instead of direct splitting
  - Extensive testing with synthetic data

**4. Quality Assessment Validity (MEDIUM RISK)**
- **Risk:** LLM assessments not correlating with actual topic quality
- **Mitigation:**
  - Combine multiple quality metrics (LLM + mathematical)
  - Human evaluation calibration
  - A/B testing against baseline methods

### Alternative Approaches (If Primary Approach Fails)

**Alternative 1: Static Topic Optimization**
- Single-pass LLM analysis without iterative refinement
- Lower cost, reduced complexity
- Still provides resilience mapping value

**Alternative 2: Human-in-the-Loop Approach**
- LLM suggestions with human approval for each iteration
- Higher quality assurance but requires human resources
- Hybrid automation with expert oversight

**Alternative 3: Rule-Based Optimization**
- Use mathematical topic quality metrics only
- Deterministic merge/split decisions based on coherence thresholds
- No API costs, fully reproducible results

### Fallback Technologies
- **Embedding Models:** `all-mpnet-base-v2` → `all-MiniLM-L6-v2` → `TF-IDF`
- **LLM Services:** OpenAI GPT-4 → GPT-3.5 → Local Llama → Rule-based
- **Topic Operations:** MERGE+SPLIT → MERGE-only → Static analysis

## Git Setup

### Repository Initialization
This project should be version controlled with Git. Initialize with:
```bash
git init
git add .
git commit -m "Initial commit: Project setup"
```

### .gitignore Configuration
Create a `.gitignore` file with the following contents:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
PIPFILE.lock

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Jupyter Notebook
.ipynb_checkpoints

# PyCharm
.idea/

# VS Code
.vscode/

# Operating System
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
results/
data/models/
*.log
.env.local
.env.development.local
.env.test.local
.env.production.local

# Large model files
*.bin
*.safetensors
models/
cache/
```

### Environment File
Create an `environment.yml` file for mamba/conda environment management:
```yaml
name: augmentation-agent-topic
channels:
  - conda-forge
  - pytorch
  - huggingface
dependencies:
  - python=3.10
  - pandas>=1.5.0
  - numpy>=1.21.0
  - scikit-learn>=1.1.0
  - nltk>=3.7
  - plotly>=5.10.0
  - jupyter>=1.0.0
  - pytest>=7.0.0
  - pip>=22.0.0
  - pip:
    - bertopic>=0.15.0
    - openai>=1.0.0
    - langchain>=0.1.0
    - tiktoken>=0.5.0
    - transformers>=4.30.0
    - torch>=2.0.0
    - sentence-transformers>=2.2.0
    - umap-learn>=0.5.3
    - hdbscan>=0.8.29
```