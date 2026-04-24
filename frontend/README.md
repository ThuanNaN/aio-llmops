# LLM Frontend Interface

A Gradio-based web interface for interacting with the routed FastAPI gateway.

## Features

- **Multiple Interfaces**: Tab-based interface for different LLM applications
- **Routed Chat**: Let the gateway classify the request and choose the best serving backend
- **Math QA**: Ask math questions pinned to the TensorRT-LLM route
- **Medical QA**: Answer free-form Vietnamese medical questions with optional context
- **Routing Metadata**: Inspect which provider, model, and route served each request
- **Examples**: Pre-populated examples for easy testing
- **Error Handling**: Robust error handling for API failures

## Architecture

The frontend communicates with the backend API to provide:

- User-friendly interface for LLM capabilities
- Form-based input collection
- Formatted result display with routing metadata
- Error reporting

## Tabs

1. **Routed Chat**: Send free-form math or medical prompts and let the gateway classify them
2. **Math QA**: Send math-only prompts to the TensorRT-LLM route
3. **Medical QA**: Submit free-form medical questions with optional patient context

## Environment Variables

Configure the service with:

- `OPENAI_API_BASE_URL`: URL of the vLLM service, defaulting to `http://192.168.1.101:8000/v1`
- `BACKEND_API_URL`: URL of the backend API service
- `OPENAI_API_KEY`: API key for authorization

## Running the Service

### Using Docker Compose

```bash
docker compose up -d
```

The Gradio interface will be available at `http://192.168.1.101:7860` in the default two-node deployment.

### Development Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables in `.env`

3. Run the application:

   ```bash
   python app.py
   ```

## Usage

### Routed Chat

1. Navigate to the "Routed Chat" tab
2. Enter a math or medical prompt
3. Optionally force a route override
4. Click "Send Request"
5. Inspect the answer and routing metadata

### Math QA

1. Navigate to the "Math QA" tab
2. Enter a math question
3. Click "Solve Math Question"
4. View the answer and serving metadata

### Medical QA

1. Navigate to the "Medical QA" tab
2. Enter a medical question
3. Optionally add symptoms, patient history, or supporting context
4. Click "Get Answer"
5. View the generated answer and serving metadata
