# Social World Model UI

A beautiful web interface for visualizing agent interactions and conversations from the Social World Model, with integrated Python API.

## Features

- **Agent Presence Timeline**: Visualize when agents are present in the conversation
- **Conversation View**: See the conversation flow between agents
- **Timeline View**: Detailed view of each timestep with state, observations, and actions
- **Python API**: FastAPI backend for the Social World Model

## Project Structure

- `/api` - Python FastAPI application
  - `/api/ai` - Social World Model API endpoints
- `/app` - Next.js application
  - `/app/api` - Next.js API routes
- `/public` - Static files

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) 18.0.0 or later
- [Python](https://python.org/) 3.10 or later
- [Bun](https://bun.sh/) 1.0.0 or later (optional)

### Installation

1. Clone the repository
2. Navigate to the UI directory:
   ```bash
   cd ui
   ```
3. Install Node.js dependencies:
   ```bash
   npm install
   # or
   bun install
   ```
4. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
5. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Development Server

```bash
npm run dev
# or
bun dev
```

This will start both the Next.js server and the FastAPI server concurrently.

The Next.js server will run on [http://localhost:3000](http://localhost:3000).
The FastAPI server will run on [http://localhost:8000](http://localhost:8000).

For allowing external access and specifying a port:

```bash
npm run next-dev -- -p 12000 -H 0.0.0.0
# or
bun dev -- -p 12000 -H 0.0.0.0
```

### Building for Production

```bash
npm run build
# or
bun run build
```

### Running in Production Mode

```bash
npm start
# or
bun start
```

## Data Format

The UI expects data in the following JSON format:

```json
{
  "agents_names": [
    "Agent1",
    "Agent2",
    "Agent3"
  ],
  "socialized_context": [
    {
      "timestep": "1",
      "state": "Description of the current state",
      "observations": [
        "Agent1: observation",
        "Agent2: observation",
        "Agent3: observation"
      ],
      "actions": [
        "Agent1: action",
        "Agent2: action",
        "Agent3: action"
      ]
    }
  ]
}
```

## Technologies Used

- Next.js
- React
- TypeScript
- Tailwind CSS
- Bun (JavaScript runtime & package manager)

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

## API Endpoints

The Python API provides the following endpoints:

- `GET /api/ai` - Welcome message
- `POST /api/ai/socialize-context` - Analyzes and socializes a context for simulation
- `POST /api/ai/initialize-simulation` - Initializes a simulation from a socialized context
- `POST /api/ai/reason-about-belief` - Reasons about an agent's belief in response to a question
- `GET /api/ai/get-simulation` - Retrieves the current state of the simulation

## Deploy on Vercel

The easiest way to deploy your Next.js app with Python API is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme).

The `vercel.json` file includes the necessary configuration to exclude Next.js files from the Python function build, preventing the "A Serverless Function has exceeded the unzipped maximum size of 250 MB" error.
