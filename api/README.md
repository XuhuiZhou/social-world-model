# Social World Model API

This API provides endpoints to interact with the Social World Model for simulating social interactions.

## Endpoints

### GET /

Welcome message to confirm the API is running.

### POST /socialize-context

Analyzes and socializes a context for simulation.

**Request Body:**
```json
{
  "context": "A narrative describing a social situation",
  "example_analysis": "Optional example analysis to guide socialization",
  "feedback": "Optional feedback for improving the context",
  "task_specific_instructions": "Optional task-specific instructions"
}
```

### POST /initialize-simulation

Initializes a simulation from a socialized context.

**Request Body:**
```json
{
  "agents_names": ["Agent1", "Agent2"],
  "socialized_context": [...],
  "context_manual": "..."
}
```

### POST /reason-about-belief

Reasons about an agent's belief in response to a question.

**Request Body:**
```json
{
  "question": "What does Agent1 think about X?",
  "agents": ["Agent1", "Agent2"],
  "target_agent": "Agent1",
  "answer_candidates": ["Option A", "Option B"]
}
```

### GET /get-simulation

Retrieves the current state of the simulation.

## Environment Variables

- `MODEL_NAME`: The name of the LLM model to use (default: "gpt-3.5-turbo")
- `TEMPERATURE`: Temperature parameter for generation (default: 0.7)