# RAG-based Legal Assistant Chatbot 🤖⚖️

A powerful, context-aware legal assistant chatbot built with LangChain and advanced RAG techniques. This application uses Retrieval Augmented Generation (RAG) with multi-query retrieval and reciprocal rank fusion to provide accurate legal information based on your documents while maintaining conversation history.

## 🌟 Features

- **PDF Document Processing**: Automatically processes and indexes legal PDF documents
- **Multi-Query Retrieval**: Generates multiple query variations for improved retrieval coverage
- **Reciprocal Rank Fusion (RRF)**: Advanced document ranking for better result quality
- **Intelligent Query Classification**: Determines whether document retrieval is needed
- **Conversation History Awareness**: Maintains context across multiple questions
- **Vector Database Storage**: Efficiently stores and retrieves document embeddings using Chroma DB
- **Command-Line Interface**: Interactive terminal-based chat interface
- **Responsible AI Disclaimers**: Clearly communicates that responses are not substitutes for legal advice

## 🛠️ Technical Stack

- **Framework**: LangChain
- **Interface**: Command-line terminal interface
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Chat Model**: Cohere (command-r model)
- **Vector Store**: ChromaDB
- **Document Processing**: LangChain's PyPDFLoader
- **Text Splitting**: RecursiveCharacterTextSplitter
- **Advanced Features**: Multi-query generation, Reciprocal Rank Fusion

## 📋 Prerequisites

```bash
python 3.12.7
pip or uv (recommended)
```

## 🚀 Installation

### Option 1: Using UV (Recommended)

1. Install UV (if not already installed):
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

2. Clone the repository:
```bash
git clone https://github.com/sougaaat/RAG-based-Legal-Assistant.git
cd RAG-based-Legal-Assistant
```

3. Install dependencies with UV:
```bash
uv venv
uv pip install -r requirements.txt
```

4. Activate the virtual environment:
```bash
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

### Option 2: Using Traditional pip

1. Clone the repository:
```bash
git clone https://github.com/sougaaat/RAG-based-Legal-Assistant.git
cd RAG-based-Legal-Assistant
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
```bash
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

4. Install required packages:
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

Set up your environment variables by editing the `.env` file:
```bash
# Edit .env with your API keys:
COHERE_API_KEY=your_cohere_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional, if using OpenAI models
```

**Note**: The `.env` file is included in the repository with empty values. Simply add your actual API keys to get started.

## 📁 Project Structure

```
├── data/                    # Store your PDF documents here
├── data-ingestion-local/    # Vector database storage (auto-created)
├── prompts/                 # Prompt templates directory
│   ├── mainRAG-prompt.md    # Main RAG system prompt
│   └── multiQuery-prompt.md # Multi-query generation prompt
├── subparts/               # Additional modules/components
├── demo/                   # Demo files and examples
├── OLD/                    # Legacy code versions
├── data-ingestion.py       # Data ingestion and vector DB setup
├── app.py                  # Main application file
├── requirements.txt        # Python dependencies
├── uv.lock                 # UV lock file for reproducible builds
├── pyproject.toml          # Project configuration
├── .env                    # Environment variables (create from .env.example)
└── README.md               # This file
```

## 🎬 Demo

Want to see the Legal Assistant in action? Navigate to the `demo/` directory and open the `DEMO.mp4` file to watch the demonstration video.

## 💫 Usage

1. **Prepare your documents**: Place your legal PDF documents in the `data/` directory

2. **Ingest documents**: Run the data ingestion script to process and index your documents:
```bash
python data-ingestion.py
```

3. **Start the chatbot**: Run the main application:
```bash
python app.py
```

4. **Interact with the bot**: The terminal interface will start, and you can begin asking questions. Type `exit` to quit.

## ⚙️ How It Works

The application uses an advanced RAG pipeline with several sophisticated components:

### 1. **Document Processing**:
   - Loads PDF documents from the data directory using PyPDFLoader
   - Splits documents into manageable chunks using RecursiveCharacterTextSplitter
   - Creates embeddings using HuggingFace's all-MiniLM-L6-v2 model
   - Stores embeddings in ChromaDB for efficient retrieval

### 2. **Intelligent Query Processing**:
   - **Query Classification**: Uses a Pydantic model to determine if document retrieval is needed
   - **Multi-Query Generation**: For complex queries, generates multiple semantically equivalent variations
   - **Context Awareness**: Incorporates chat history for contextual understanding

### 3. **Advanced Retrieval System**:
   - **Multi-Query Retrieval**: Searches the knowledge base with multiple query variations
   - **Reciprocal Rank Fusion (RRF)**: Combines and ranks results from different queries using RRF algorithm
   - **Smart Document Selection**: Returns the top 3 most relevant documents based on RRF scores

### 4. **Response Generation**:
   - Uses Cohere's language model for generating responses
   - Provides clear, concise answers based on retrieved context
   - Maintains conversation history for follow-up questions
   - Includes appropriate legal disclaimers

## 🔧 Key Components

- **MultiQuery Class**: Pydantic model for structured query analysis and multi-query generation
- **createMultiQueryChain()**: Creates the multi-query generation chain
- **generateRRF()**: Implements Reciprocal Rank Fusion for document ranking
- **generateResponse()**: Main response generation function with context integration

## ⚠️ Important Notes

- This chatbot is designed to provide information only and should not be used as a substitute for professional legal advice
- Responses are generated based on the provided document context and conversation history
- The system intelligently determines when document retrieval is necessary vs. conversational responses
- All responses include appropriate disclaimers about the limitations of AI-generated legal information

## 🔑 API Keys Required

- **Cohere API Key**: Required for the chat model (sign up at [Cohere](https://cohere.com/))
- **OpenAI API Key**: Optional, if you want to switch to OpenAI models

## 👨‍💻 Creator

Created by [Sougat Dey](https://www.linkedin.com/in/sougatdey/)

## 📄 License

© 2024 Sougat Dey. All rights reserved.

This project is not licensed under any open-source license. You may not copy, distribute, or modify this project without permission.

Without a license, all rights are reserved to the creator. If you are interested in using this code or contributing, please contact me for permission or consider using an appropriate open-source license.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/sougaaat/RAG-based-Legal-Assistant/issues).

## 🌟 Show your support

Give a ⭐️ if this project helped you!