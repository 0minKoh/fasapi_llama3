# Supaja AI Chatbot - LLM Server (Ollama Integration)

This repository contains the FastAPI-based AI server for the Supaja AI Chatbot project. It acts as a proxy, communicating with a locally running Ollama instance to perform Large Language Model (LLM) inference using the Llama 3.1 8B model.

This server is designed to be run on a separate machine or process from the main Django web server, enabling efficient and scalable LLM inference.

## 🚀 Key Features

- **Ollama Integration**: Leverages Ollama for easy local LLM deployment and efficient inference, especially optimized for Apple Silicon (M-series Macs).
- **FastAPI**: Provides a high-performance, asynchronous web API for LLM inference requests.
- **Llama 3.1 8B**: Utilizes the powerful Llama 3.1 8B Instruct model for natural language understanding and generation.
- **Clean API**: Offers a simple `/generate/` endpoint for sending prompts and receiving LLM responses.
- **Automatic Docs**: FastAPI automatically generates interactive API documentation (Swagger UI).

## 🛠️ Setup and Installation

### Prerequisites

Before you can run this LLM server, you **must** have Ollama installed and the `llama3` model pulled on your system.

1.  **Install Ollama**:

    - Visit the [Ollama website](https://ollama.com/download) and download the installer for macOS (or your operating system). Follow the installation instructions.

2.  **Pull Llama 3 Model**:

    - Open your terminal and pull the Llama 3 8B model. This model is approximately 4.7 GB.
      ```bash
      ollama pull llama3
      ```
    - (Optional) To test if Ollama is working, you can run: `ollama run llama3` and chat with the model directly. Type `/bye` to exit.

3.  **Verify Ollama API Server**:
    - Ollama automatically runs an API server on `http://localhost:11434`. You can test it with `curl`:
      ```bash
      curl http://localhost:11434/api/chat -d '{
        "model": "llama3",
        "messages": [
          {
            "role": "user",
            "content": "What is the capital of France?"
          }
        ],
        "stream": false
      }'
      ```
      You should receive a JSON response with the model's answer.

### Server Installation

1.  **Navigate to `ai_server` Directory**:
    Open your terminal and change to the `ai_server` directory.

    ```bash
    cd path/to/your/ai_chatbot_project/ai_server
    ```

2.  **Create and Activate a Virtual Environment**:
    It's highly recommended to use a virtual environment to manage dependencies.

    ```bash
    python -m venv venv_ai
    # macOS/Linux
    source venv_ai/bin/activate
    # Windows
    .\venv_ai\Scripts\activate
    ```

3.  **Install Python Dependencies**:
    ```bash
    pip install fastapi uvicorn requests
    ```

## 🚀 Running the LLM Server

1.  **Ensure Ollama is Running**:
    Make sure the Ollama application is running in the background. If you restart your machine, you might need to launch the Ollama application again.

2.  **Start the FastAPI Server**:
    With your `venv_ai` virtual environment activated, run the server:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload
    ```
    - The server will typically run on `http://127.0.0.1:8001`.
    - `--reload` is useful for development as it automatically restarts the server on code changes. Remove it for production deployments.

## 🧪 Testing the API

Once the FastAPI server is running, you can test its `/generate/` endpoint:

1.  **Access Swagger UI**:
    Open your web browser and go to `http://127.0.0.1:8001/docs`.
    This page provides an interactive API documentation.

2.  **Test the `/generate/` Endpoint**:
    - Expand the `/generate/` POST endpoint.
    - Click on "Try it out".
    - In the `prompt` field, enter a question (e.g., `"Explain RAG in simple terms."`).
    - Click "Execute".
    - You should see the LLM's generated response in the "Response body" section.

## 📂 Project Structure

```
ai_server/
├── main.py             # FastAPI application and LLM inference logic
├── venv_ai/            # Python virtual environment (ignored by Git)
├── README.md           # This file
└── .gitignore          # Git ignore file for Python environments
```

## License

[Choose a license, e.g., MIT License, Apache 2.0 License]
(Provide a link to the full license text, e.g., for MIT: [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT))

## Contributing

We welcome contributions! Please feel free to open issues or submit pull requests.
