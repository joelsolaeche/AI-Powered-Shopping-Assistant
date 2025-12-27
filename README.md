# 🛒 AI-Powered Shopping Assistant

<div align="center">

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Tests](https://img.shields.io/badge/tests-26%2F26%20passing-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**A production-ready conversational AI system for intelligent e-commerce customer service**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Contributing](#-contributing)

English | [日本語](README.ja.md)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Technologies](#-technologies)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Testing](#-testing)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This project implements an intelligent shopping assistant that revolutionizes e-commerce customer service using cutting-edge AI technologies. Built with **LangGraph**, **LangChain**, and **Model Context Protocol (MCP)**, it provides a seamless conversational experience for product discovery, cart management, and customer support.

### The Problem

Traditional e-commerce platforms struggle with:
- Generic search results that don't understand natural language
- Disconnected shopping experiences across product discovery and purchase
- Overwhelming customer support teams with routine inquiries
- Lack of personalization based on purchase history

### The Solution

Our AI Shopping Assistant provides:
- **Natural language understanding** for intuitive product search
- **Contextual conversations** that remember your shopping session
- **Intelligent escalation** to human support when needed
- **Real-time web integration** for product information and trends
- **Purchase history analysis** for personalized recommendations

---

## ✨ Features

### 🔍 Intelligent Product Search

- **Semantic Search**: Understands natural language queries like "healthy breakfast options"
- **Vector Embeddings**: Uses HuggingFace mxbai-embed-large-v1 for 49,688 products
- **Structured Filtering**: SQL-like queries for precise results (department, aisle, price)
- **Purchase History**: Analyzes your past orders for personalized suggestions

### 🛒 Smart Shopping Cart

- **Quantity Management**: Add, remove, update quantities
- **Session Persistence**: Cart maintained throughout conversation
- **Real-time Updates**: Instant feedback on cart operations
- **Checkout Flow**: Complete purchase workflow

### 🌐 Web Search Integration (MCP)

- **6 Brave Search Tools**:
  - General web search
  - News articles
  - Video content
  - Image search
  - Local results
  - AI summaries
- **Hybrid Queries**: Combines product catalog with real-time web data

### 🤝 Customer Support

- **Automatic Escalation**: Detects refund requests, complaints, account issues
- **Human-in-the-Loop**: Requires supervisor approval for sensitive operations
- **Context Preservation**: Full conversation history available to support agents

### 📊 Analytics & Insights

- **Purchase Patterns**: Most bought items, reorder frequency
- **Product Recommendations**: Based on order history
- **Trend Analysis**: Combines web search with catalog data

---

## 🎬 Demo

### Example Conversations

**Product Discovery**
```
👤 User: "I need healthy breakfast options"
🤖 Assistant: Here are some healthy breakfast items from our catalog:
   • Organic Granola (ID: 18872) - Cereal aisle
   • Greek Yogurt (ID: 24531) - Dairy aisle
   • Fresh Blueberries (ID: 13176) - Fresh fruits aisle
   Would you like to add any to your cart?
```

**Cart Management**
```
👤 User: "Add 2 boxes of granola to my cart"
🤖 Assistant: Added 2 of Organic Granola (ID: 18872) to your cart.

👤 User: "Show my cart"
🤖 Assistant: Your cart contains:
   • Organic Granola (ID: 18872) × 2
   Total items: 2
```

**Web Search + Product Catalog**
```
👤 User: "What are the health benefits of quinoa? Do we have any?"
🤖 Assistant: [Searches web] Quinoa is a complete protein containing all 9 
   essential amino acids, high in fiber, and gluten-free...
   
   [Searches catalog] Yes! We have:
   • Organic Quinoa (ID: 39187) - Grains aisle
   • Tri-Color Quinoa (ID: 42156) - Grains aisle
```

**Customer Support**
```
👤 User: "I want a refund for my last order"
🤖 Assistant: I'll connect you with customer support regarding your refund.
   [Escalates to support agent]
   
🎧 Support: I see you need help with a refund. Let me get supervisor approval...
   [Requests human approval]
   
👨‍💼 Supervisor: [Approves refund]
🎧 Support: Your refund has been approved and will be processed within 3-5 days.
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                    (Streamlit Web App)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph State Machine                      │
│              (Conversation Flow Orchestration)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   AI Agent (GPT-4o-mini)                        │
│              (Natural Language Understanding)                   │
└──────┬──────────────────┬──────────────────┬───────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐  ┌─────────────┐  ┌──────────────────┐
│   Product   │  │   Shopping  │  │   Web Search     │
│   Search    │  │    Cart     │  │   (MCP/Brave)    │
└─────────────┘  └─────────────┘  └──────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐  ┌─────────────┐  ┌──────────────────┐
│  Vector DB  │  │  In-Memory  │  │  Brave Search    │
│  (Chroma)   │  │   Storage   │  │      API         │
│             │  │             │  │                  │
│ 49,688      │  │ Session-    │  │ Real-time Web    │
│ Products    │  │ Based       │  │ Data             │
└─────────────┘  └─────────────┘  └──────────────────┘
```

### Key Components

1. **LangGraph State Machine**: Manages conversation flow and state transitions
2. **AI Agent**: GPT-4o-mini for natural language understanding and generation
3. **Tool Layer**: Modular tools for search, cart, and escalation
4. **Data Layer**: Vector DB (Chroma) + structured data (Pandas)
5. **MCP Integration**: External tool protocol for web search

---

## 🛠️ Technologies

### Core Framework
- **[LangChain](https://python.langchain.com/)** - LLM integration and tool management
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** - Stateful conversation orchestration
- **[OpenAI GPT-4o-mini](https://openai.com/)** - Natural language processing

### Data & Search
- **[Chroma](https://www.trychroma.com/)** - Vector database for semantic search
- **[HuggingFace Transformers](https://huggingface.co/)** - Text embeddings (mxbai-embed-large-v1)
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation and filtering

### Integration
- **[Model Context Protocol (MCP)](https://modelcontextprotocol.io/)** - External tool integration
- **[Brave Search API](https://brave.com/search/api/)** - Web search capabilities

### Development
- **[Pydantic](https://pydantic.dev/)** - Data validation and schema definition
- **[Pytest](https://pytest.org/)** - Testing framework
- **[Streamlit](https://streamlit.io/)** - Web interface
- **[Python-dotenv](https://pypi.org/project/python-dotenv/)** - Environment management

---

## 📦 Installation

### Prerequisites

- **Python 3.10+** ([Download](https://www.python.org/downloads/))
- **Node.js** ([Download](https://nodejs.org/)) - Required for MCP servers
- **OpenAI API Key** ([Get one](https://platform.openai.com/api-keys))
- **Brave Search API Key** ([Get one](https://brave.com/search/api/)) - Optional

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/ai-shopping-assistant.git
cd ai-shopping-assistant
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional (for web search)
BRAVE_API_KEY=your-brave-api-key-here
```

### Step 5: Download Dataset

The dataset is **not included** in the repository due to size (~50MB). Download it automatically:

```bash
python download_dataset.py
```

This will:
- Download the grocery store dataset from Google Drive
- Extract it to the `dataset/` folder
- Verify all required CSV files are present

**Dataset Contents:**
- `products.csv` - 49,688 products with names, departments, aisles
- `orders.csv` - 3.4M+ order records
- `order_products__prior.csv` - Purchase history
- `order_products__train.csv` - Training data
- `departments.csv` - 21 departments
- `aisles.csv` - 134 aisles

### Step 6: Build Vector Database

The vector database is also **not included** due to size. Build it locally:

#### Option A: Google Colab (Recommended - 10x faster)

1. Open [Google Colab](https://colab.research.google.com)
2. Enable GPU: `Runtime` → `Change runtime type` → `T4 GPU`
3. Run:

```python
!git clone https://github.com/yourusername/ai-shopping-assistant.git
%cd ai-shopping-assistant
!pip install -r requirements.txt
!python download_dataset.py
!python src/build_vector_db.py
```

4. Download the `vector_db/` folder
5. Place it in your local project root

**Time: ~5-10 minutes with GPU**

#### Option B: Local Machine

```bash
python src/build_vector_db.py
```

**Time: ~1-2 hours on CPU**

### Step 7: Verify Installation

```bash
# Run tests
pytest tests/ -v

# Should show: 26 passed
```

---

## 🚀 Usage

### Web Interface (Recommended)

Start the Streamlit app:

```bash
streamlit run app.py
```

Open your browser to: **http://localhost:8501**

### Command Line Interface

```python
from src.conversation_runner import run_single_turn

# Single turn conversation
result = run_single_turn(
    user_input="I need healthy snacks",
    thread_id="my-session-123"
)

print(result['response'])
```

### Programmatic Usage

```python
from src.graph import graph
from langchain_core.messages import HumanMessage

# Multi-turn conversation
config = {"configurable": {"thread_id": "session-456"}}

# First message
state1 = graph.invoke(
    {"messages": [HumanMessage(content="Show me organic products")]},
    config
)

# Follow-up (maintains context)
state2 = graph.invoke(
    {"messages": [HumanMessage(content="Add the first one to cart")]},
    config
)
```

---

## 📁 Project Structure

```
ai-shopping-assistant/
│
├── src/                          # Source code
│   ├── assistants.py             # AI agent implementations (sales & support)
│   ├── tools.py                  # Tool definitions (search, cart, escalation)
│   ├── graph.py                  # LangGraph conversation flow
│   ├── state.py                  # State management schemas
│   ├── prompts.py                # AI prompts and instructions
│   ├── web_search_mcp.py         # Brave Search MCP integration
│   ├── build_vector_db.py        # Vector database builder
│   ├── conversation_runner.py    # Testing utilities
│   └── __init__.py
│
├── tests/                        # Test suite (100% coverage)
│   ├── test_cart_and_schema.py   # Cart and schema validation tests
│   ├── test_end_to_end.py        # Integration tests
│   ├── test_graph.py             # Graph flow tests
│   ├── test_sales_assistant.py   # Sales agent tests
│   ├── test_structured_search.py # Structured search tests
│   ├── test_tool_node.py         # Tool execution tests
│   ├── test_vector_search.py     # Vector search tests
│   ├── test_web_search_mcp.py    # MCP integration tests
│   └── __init__.py
│
├── dataset/                      # Product catalog (downloaded)
│   ├── products.csv              # 49,688 products
│   ├── orders.csv                # Order history
│   ├── order_products__prior.csv # Purchase records
│   ├── order_products__train.csv # Training data
│   ├── departments.csv           # 21 departments
│   └── aisles.csv                # 134 aisles
│
├── vector_db/                    # Chroma vector database (built locally)
│   └── [generated files]
│
├── app.py                        # Streamlit web interface
├── requirements.txt              # Python dependencies
├── download_dataset.py           # Dataset downloader script
├── .env                          # Environment variables (not in repo)
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
├── ASSIGNMENT.md                 # Original assignment instructions
└── LICENSE                       # MIT License
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

**Expected Output:**
```
26 passed in ~20s
```

### Run Specific Test Suites

```bash
# Vector search tests
pytest tests/test_vector_search.py -v

# Structured search tests
pytest tests/test_structured_search.py -v

# Cart functionality tests
pytest tests/test_cart_and_schema.py -v

# MCP integration tests
pytest tests/test_web_search_mcp.py -v

# End-to-end workflow tests
pytest tests/test_end_to_end.py -v
```

### Test Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

Open `htmlcov/index.html` to view detailed coverage report.

**Current Coverage: 100%**

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | OpenAI API key for GPT-4o-mini |
| `BRAVE_API_KEY` | ❌ Optional | Brave Search API key for web search |

### Model Configuration

Edit `src/assistants.py` to change the LLM:

```python
# Current: GPT-4o-mini
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Alternative: GPT-4
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Alternative: GPT-3.5-turbo
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
```

### Vector Database Configuration

Edit `src/build_vector_db.py`:

```python
# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1",
    model_kwargs={"device": "cpu"},  # Change to "cuda" for GPU
    encode_kwargs={"normalize_embeddings": True}
)

# Batch size (adjust based on memory)
batch_size = 256  # Increase for more memory, decrease for less
```

---

## 📚 API Documentation

### Core Functions

#### `search_products(query: str, top_k: int = 5)`

Performs semantic vector search over the product catalog.

**Parameters:**
- `query` (str): Natural language search query
- `top_k` (int): Number of results to return

**Returns:**
- List of dicts with product_id, product_name, aisle, department, text

**Example:**
```python
from src.tools import search_products

results = search_products("healthy snacks", top_k=5)
for product in results:
    print(f"{product['product_name']} - {product['department']}")
```

#### `structured_search_tool(...)`

SQL-like filtering over the product catalog.

**Parameters:**
- `product_name` (str, optional): Substring match
- `department` (str, optional): Exact department match
- `aisle` (str, optional): Aisle name match
- `history_only` (bool): Filter by user's purchase history
- `reordered` (bool): Only reordered items
- `min_orders` (int): Minimum order count
- `order_by` (str): Sort by "count" or "add_to_cart_order"
- `top_k` (int): Limit results

**Example:**
```python
from src.tools import structured_search_tool

# Find organic products in produce department
results = structured_search_tool.invoke({
    "product_name": "organic",
    "department": "produce",
    "top_k": 10
})
```

#### `cart_tool(cart_operation: str, product_id: int, quantity: int = 1)`

Manages shopping cart operations.

**Parameters:**
- `cart_operation` (str): "add", "remove", "update", or "buy"
- `product_id` (int): Product ID from search results
- `quantity` (int): Quantity to add/remove/update

**Example:**
```python
from src.tools import cart_tool, view_cart

# Add product
cart_tool.invoke({"cart_operation": "add", "product_id": 24852, "quantity": 2})

# View cart
cart_contents = view_cart.invoke({})
print(cart_contents)
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "No module named 'langchain'"

**Solution:**
```bash
pip install -r requirements.txt
```

#### 2. "OPENAI_API_KEY not found"

**Solution:**
- Create `.env` file in project root
- Add: `OPENAI_API_KEY=your-key-here`
- Restart the application

#### 3. "Vector database not found"

**Solution:**
```bash
python src/build_vector_db.py
```

#### 4. "Dataset files missing"

**Solution:**
```bash
python download_dataset.py
```

#### 5. "Tests failing"

**Solution:**
```bash
# Ensure vector_db exists
python src/build_vector_db.py

# Ensure dataset exists
python download_dataset.py

# Run tests
pytest tests/ -v
```

#### 6. "MCP tools not loading"

**Solution:**
- Check Node.js is installed: `node --version`
- Verify BRAVE_API_KEY in `.env`
- Check internet connection

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Style

We use [Black](https://black.readthedocs.io/) for code formatting:

```bash
black --line-length=88 .
```

### Testing Requirements

- All new features must include tests
- Maintain 100% test coverage
- All tests must pass before PR approval

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new search filter
fix: resolve cart quantity bug
docs: update README installation steps
test: add vector search tests
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Dataset
- **Instacart Market Basket Analysis** - Grocery store dataset with 49K+ products

### Technologies
- **LangChain & LangGraph** - Conversation orchestration framework
- **OpenAI** - GPT models for natural language understanding
- **Brave Search** - Web search API and MCP integration
- **HuggingFace** - Text embedding models
- **Chroma** - Vector database

### Inspiration
- This project was developed as part of an LLM Agents course
- Special thanks to the course instructors and community

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-shopping-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-shopping-assistant/discussions)
- **Email**: joel_solaeche@hotmail.com

---

## 🗺️ Roadmap

### Current Version (v1.0)
- ✅ Semantic product search
- ✅ Shopping cart management
- ✅ Customer support escalation
- ✅ Web search integration (MCP)
- ✅ 100% test coverage

### Future Enhancements (v2.0)
- [ ] Multi-language support
- [ ] Voice interface
- [ ] Product recommendations ML model
- [ ] Order tracking integration
- [ ] Payment processing
- [ ] Mobile app
- [ ] Analytics dashboard

---

<div align="center">

**Built with ❤️ using LangGraph, LangChain, and MCP**

⭐ Star this repo if you find it helpful!

[Report Bug](https://github.com/yourusername/ai-shopping-assistant/issues) • [Request Feature](https://github.com/yourusername/ai-shopping-assistant/issues)

</div>
