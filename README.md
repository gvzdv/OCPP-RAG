# RAG Application Chatbot Prototype

## Architecture
- **LLM**: Cohere Command R+
- **Embedding model**: embed-english-v3.0
- **Reranking model**: rerank-english-v3.0

## Data
- OCPP 1.6 (HTML format)
- WebSocket API documentation
- Any HTML-based document can be added based on the application purpose

## What this application does

Command R+ model is designed with RAG (Retrieval-Augmented Generation) in mind. 
This application is built to parse information from OCPP 1.6, WebSocket documentation, and essentially any documents available online and precisely answer user's questions.

Based on my tests, it performs better than a ChatGPT-based chatbot offered at the [Open Charge Alliance website](https://openchargealliance.org/oca-i-chatbot/).

## Examples

### Example 1

**Official chatbot:**
![Official chatbot response 1](img.png)

**This application:**
![Our application response 1](img_2.png)

**Correct answer:**
![Correct answer 1](img_3.png)

### Example 2

**Official chatbot:**
![Official chatbot response 2](img_1.png)

**This application:**
![Our application response 2](img_4.png)

**Correct answer:**
![Correct answer 2](img_5.png)