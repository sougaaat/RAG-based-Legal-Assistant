# RAG-based Legal Assistant Chatbot 🤖⚖️

A powerful, context-aware legal assistant chatbot built with LangChain and Streamlit. This application uses Retrieval Augmented Generation (RAG) to provide accurate legal information based on your documents while maintaining conversation history.

## ▶️ Demo

<video width="600" controls>
  <source src="demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## 🌟 Features

- **PDF Document Processing**: Automatically processes and indexes legal PDF documents
- **Intelligent Retrieval**: Uses semantic search to find relevant information from your legal documents
- **Conversation History Awareness**: Maintains context across multiple questions
- **Vector Database Storage**: Efficiently stores and retrieves document embeddings using Chroma DB
- **User-Friendly Interface**: Clean and intuitive chat interface built with Streamlit
- **Responsible AI Disclaimers**: Clearly communicates that responses are not substitutes for legal advice

## 🛠️ Technical Stack

- **Framework**: LangChain
- **Frontend**: Streamlit
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Chat Model**: llama-3.1-8b-instant
- **Vector Store**: ChromaDB
- **Document Processing**: LangChain's PyPDFLoader
- **Text Splitting**: RecursiveCharacterTextSplitter

## 📋 Prerequisites

```bash
python 3.12.7
pip
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/sougaaat/RAG-based-Legal-Assistant.git
cd RAG-based-Legal-Assistant
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## 📁 Project Structure

```
├── data/                  # Store your PDF documents here
├── data-ingestion-local/  # Vector database storage
├── data-ingestion.py      # Data Ingestion script
├── main-app.py            # Main application file
├── .env                   # Environment variables
└── README.md
```

## 💫 Usage

1. Place your legal PDF documents in the `data/` directory

2. Run the application:
```bash
python data-ingestion.py
streamlit run main-app.py
```

3. Access the web interface at `http://localhost:8501`

## ⚙️ How It Works

1. **Document Processing**:
   - Loads PDF documents from the data directory
   - Splits documents into manageable chunks
   - Creates embeddings using HuggingFace's model

2. **Query Processing**:
   - Reformulates user queries to maintain context based on chat history
   - Retrieves relevant document chunks
   - Generates concise, accurate responses

3. **Response Generation**:
   - Provides clear, concise answers (maximum 4 sentences)
   - Includes legal disclaimer with every response
   - Maintains conversation history for context

## ⚠️ Important Notes

- This chatbot is designed to provide information only and should not be used as a substitute for professional legal advice
- Responses are generated based on the provided document context
- The system will explicitly state when it cannot provide information about a topic

## 👨‍💻 Creator

Created by [Sougat Dey](https://www.linkedin.com/in/sougatdey/)

## 📄 License

© 2024 Your Name. All rights reserved.

This project is not licensed under any open-source license. You may not copy, distribute, or modify this project without permission.

Without a license, all rights are reserved to the creator. If you are interested in using this code or contributing, please contact me for permission or consider using an appropriate open-source license.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/sougaaat/RAG-based-Legal-Assistant/issues).

## 🌟 Show your support

Give a ⭐️ if this project helped you!
