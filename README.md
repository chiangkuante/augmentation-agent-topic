# Augmentation Agent Topic

🤖 **LLM-assisted iterative topic modeling framework** for digital resilience analysis of corporate reports.

## 🎯 Overview

This system combines **BERTopic** with **LLM intelligence** to automatically optimize topic models and generate digital resilience scores for companies. It analyzes corporate reports to derive insights about organizational resilience across three key dimensions:

- **🛡️ Absorption Capacity**: Defense and stability capabilities
- **🔄 Adaptive Capacity**: Adjustment and flexibility capabilities
- **🚀 Transformation Capacity**: Revolutionary change and evolution capabilities

## ⚡ Quick Start

### 1. Setup Environment

```bash
# Clone and navigate to project
git clone <repository-url>
cd augmentation-agent-topic

# Run automated setup
python setup_environment.py
```

The setup script will:
- ✅ Check Python 3.10+ requirement
- ✅ Validate dependencies
- ✅ Create `.env` configuration file
- ✅ Set up OpenAI API key
- ✅ Create necessary directories

### 2. Install Dependencies

**📚 For comprehensive installation instructions and troubleshooting, see: [INSTALLATION.md](INSTALLATION.md)**

```bash
# Using mamba (recommended)
mamba env create -f mamba-environment.yml
mamba activate augmentation-agent-topic

# Or using conda
conda env create -f environment.yml
conda activate augmentation-agent-topic

# Or using pip
pip install -r requirements.txt
```

**💡 Having installation issues?** Check the [troubleshooting guide](INSTALLATION.md#-troubleshooting) for solutions to common problems including:
- Mamba/conda package conflicts
- PyTorch installation issues
- BERTopic dependencies
- Environment variable loading

### 3. Configure Environment

Choose your environment:
- **Development**: `cp .env.development .env` (testing with small budget)
- **Production**: `cp .env.production .env` (full runs with higher budget)

Edit `.env` and set your OpenAI API key:
```bash
OPENAI_API_KEY=your-actual-api-key-here
```

### 4. Add Your Data

Place your corpus file in the `data/` directory:
```bash
cp your_corpus_file.csv data/corpus_semantic_chunks.csv
```

Expected CSV format:
- **ticker**: Company symbol
- **year**: Report year
- **text**: Document text content

### 5. Run the Pipeline

```bash
# Validate setup
python src/main.py --validate-config

# Quick test run
python src/main.py --sample-size 100 --budget 5.0

# Full production run
python src/main.py --budget 50.0
```

## 🏗️ System Architecture

```
📊 Data Loading → 🤖 Topic Modeling → 🧠 LLM Analysis → 🔄 Optimization → 📈 Resilience Scoring
```

### Core Components

1. **📁 Data Loader** (`src/data_loader.py`)
   - Multi-encoding CSV support
   - Data validation and cleaning
   - Statistical analysis

2. **🎯 Topic Modeler** (`src/initial_topic_modeler.py`)
   - BERTopic with sentence-transformers
   - UMAP + HDBSCAN clustering
   - Quality metrics calculation

3. **🧠 LLM Agent** (`src/llm_agent.py`)
   - OpenAI GPT-4/3.5 integration
   - **Hard budget limits** with automatic stopping
   - Batch processing for cost optimization
   - Chain-of-Thought reasoning

4. **🔄 Optimizer** (`src/iterative_optimizer.py`)
   - **Phase 1**: MERGE-only operations (stable MVP)
   - Quality-based convergence detection
   - Automatic rollback on degradation
   - Comprehensive checkpointing

5. **📊 Resilience Mapper** (`src/resilience_mapper.py`)
   - Digital resilience framework mapping
   - Company-year score calculation
   - Confidence metrics

## 🛠️ Configuration

The system uses **dotenv** for comprehensive environment management:

### Environment Variables

```bash
# API Configuration
OPENAI_API_KEY=your-key-here
MAX_API_BUDGET=50.0
DEFAULT_LLM_MODEL=gpt-5

# Optimization Settings
MAX_OPTIMIZATION_ITERATIONS=5
CONVERGENCE_THRESHOLD=0.02
ENABLE_TOPIC_SPLIT=false

# Topic Modeling
MIN_TOPIC_SIZE=10
DEFAULT_EMBEDDING_PHASE=phase1
USE_GPU=false

# Resilience Analysis
RESILIENCE_CONFIDENCE_THRESHOLD=0.3
RESILIENCE_SCORING_METHOD=weighted_average
```

### Configuration Commands

```bash
# Show current configuration
python src/main.py --show-config

# Validate configuration
python src/main.py --validate-config

# Switch environments
python src/main.py --env production
```

## 💰 Cost Management

The system includes comprehensive cost controls:

- **💵 Hard Budget Limits**: Automatic stopping when budget exceeded
- **📊 Real-time Tracking**: Monitor API costs during execution
- **🎛️ Batch Processing**: Reduce API calls by 60-80%
- **⚠️ Budget Alerts**: Warnings at configurable thresholds

Example cost-controlled run:
```bash
python src/main.py --model gpt-5-nano-2025-08-07 --max-iterations 3
```

## 🎚️ Usage Examples

### Development Testing
```bash
# Small sample with low budget
python src/main.py \
  --env development \
  --sample-size 100 \
  --budget 5.0 \
  --max-iterations 2
```

### Production Analysis
```bash
# Full corpus analysis
python src/main.py \
  --env production \
  --budget 100.0 \
  --max-iterations 10 \
  --model gpt-5
```

### Custom Configuration
```bash
# Cost-optimized run
python src/main.py \
  --model gpt-5-nano-2025-08-07 \
  --budget 25.0 \
  --convergence-threshold 0.05 \
  --skip-optimization
```

## 📈 Output Results

The system generates comprehensive results in `results/`:

```
results/
├── pipeline/
│   └── run_YYYYMMDD_HHMMSS/
│       ├── pipeline_summary.json      # Executive summary
│       ├── topics.csv                 # Final topics
│       ├── topic_model.pkl           # Trained model
│       └── resilience/
│           ├── resilience_scores.csv  # Company scores
│           └── topic_resilience_mappings.csv
├── optimization/                     # Optimization history
└── checkpoints/                     # Intermediate results
```

### Key Outputs

1. **📊 Resilience Scores**: Company-year resilience metrics
2. **🎯 Optimized Topics**: LLM-enhanced topic models
3. **💰 Cost Reports**: Detailed API usage and costs
4. **📈 Quality Metrics**: Model performance tracking
5. **🔍 Audit Trail**: Complete optimization history

## 🚀 Advanced Features

### Multi-Environment Support
- **Development**: Low-cost testing environment
- **Production**: Full-feature deployment
- **Custom**: Tailored configurations

### Phase-Based Implementation
- **Phase 1 (MVP)**: MERGE-only operations, proven stability
- **Phase 2**: SPLIT functionality, ModernBERT integration
- **Phase 3**: Advanced features, local LLM support

### Quality Assurance
- **📊 Multi-metric Assessment**: Coherence, silhouette, diversity
- **🔄 Automatic Rollback**: Revert to best previous state
- **✅ Convergence Detection**: Smart stopping criteria
- **🎯 Human Validation**: Expert evaluation hooks

## 🛡️ Safety & Reliability

- **🔒 API Key Validation**: Secure credential management
- **💾 Comprehensive Checkpointing**: Never lose progress
- **⚡ Graceful Error Handling**: Robust failure recovery
- **📝 Detailed Logging**: Full audit trails
- **🎛️ Resource Management**: Memory and GPU optimization

## 📚 Documentation

- **`CLAUDE.md`**: Complete technical specifications
- **`todo.md`**: Development roadmap and tasks
- **`note.md`**: System architecture (Traditional Chinese)
- **`.env.example`**: Complete configuration template

## 🤝 Contributing

The system is designed for extensibility:

1. **Add New Embedding Models**: Update `EMBEDDING_MODELS` configuration
2. **Extend Resilience Framework**: Modify resilience dimensions
3. **Add New LLM Providers**: Implement in `llm_agent.py`
4. **Custom Quality Metrics**: Extend quality assessment

## ⚠️ Requirements

- **Python**: 3.10+
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 5-10GB for models and results
- **API Budget**: $50-100 recommended per full run

## 📄 License

This project is part of a research initiative for digital resilience analysis.

---

🤖 **Built with Claude Code** - Combining the power of traditional ML with modern LLM intelligence for enterprise-grade topic modeling and resilience analysis.