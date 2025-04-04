# Social World Model with Vercel Python Functions

This branch adds Vercel Functions support to the Social World Model project, allowing you to deploy the SocialWorldModel class as a serverless API.

## Project Structure

The project is organized into two main components:

1. **Core Social World Model** - The main Python library for simulating social interactions
2. **UI with Python API** - A Next.js frontend with integrated Python API using Vercel Functions

All Vercel-related files are contained within the `ui` folder:

```
social-world-model/
├── social_world_model/     # Core Python library
└── ui/                     # UI with Python API
    ├── api/                # Python FastAPI application
    │   └── ai/             # Social World Model API endpoints
    ├── app/                # Next.js application
    │   └── api/            # Next.js API routes
    ├── public/             # Static files
    ├── vercel.json         # Vercel configuration
    ├── next.config.js      # Next.js configuration
    ├── package.json        # Node.js dependencies
    ├── .env                # Environment variables
    └── requirements.txt    # Python dependencies
```

## Implementation Details

### Python API

The Python API is implemented using FastAPI and is located in the `ui/api` directory. It provides endpoints for interacting with the Social World Model:

- `GET /api/ai` - Welcome message
- `POST /api/ai/socialize-context` - Analyzes and socializes a context for simulation
- `POST /api/ai/initialize-simulation` - Initializes a simulation from a socialized context
- `POST /api/ai/reason-about-belief` - Reasons about an agent's belief in response to a question
- `GET /api/ai/get-simulation` - Retrieves the current state of the simulation

### Next.js Frontend

The Next.js frontend is located in the `ui/app` directory and provides a user interface for interacting with the Social World Model.

### Vercel Configuration

The `ui/vercel.json` file includes the necessary configuration to deploy both the Next.js frontend and the Python API to Vercel. It includes the following key configurations:

1. **Functions Configuration** - Excludes Next.js files from the Python function build to prevent size limit errors
2. **Builds Configuration** - Specifies how to build the Python API and Next.js frontend
3. **Routes Configuration** - Defines how requests are routed to the appropriate handlers

## Local Development

1. Navigate to the UI directory:
   ```bash
   cd ui
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Node.js dependencies:
   ```bash
   npm install
   ```

5. Run the development server:
   ```bash
   npm run dev
   ```

This will start both the Next.js server and the FastAPI server concurrently.

## Deployment to Vercel

1. Push your code to a Git repository
2. Connect the repository to Vercel
3. Deploy

The `vercel.json` file includes the necessary configuration to exclude Next.js files from the Python function build, preventing the "A Serverless Function has exceeded the unzipped maximum size of 250 MB" error.