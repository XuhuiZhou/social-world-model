# Social World Model with Vercel Python Functions

This project integrates the Social World Model with Vercel's Python Functions and Next.js.

## Project Structure

- `/api/index.py` - The main FastAPI application that exposes the SocialWorldModel functionality
- `/app` - Next.js application
- `/public` - Static files
- `/vercel.json` - Configuration for Vercel deployment
- `/next.config.js` - Next.js configuration with rewrites for local development
- `/requirements.txt` - Python dependencies for Vercel Python runtime
- `/package.json` - Node.js dependencies and scripts

## Local Development

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Node.js dependencies:
   ```bash
   npm install
   ```

4. Run the development server:
   ```bash
   npm run dev
   ```

This will start both the Next.js server and the FastAPI server concurrently.

## API Endpoints

- `GET /api/ai` - Welcome message
- `POST /api/ai/socialize-context` - Analyzes and socializes a context for simulation
- `POST /api/ai/initialize-simulation` - Initializes a simulation from a socialized context
- `POST /api/ai/reason-about-belief` - Reasons about an agent's belief in response to a question
- `GET /api/ai/get-simulation` - Retrieves the current state of the simulation

## Deployment to Vercel

1. Push your code to a Git repository
2. Connect the repository to Vercel
3. Deploy

The `vercel.json` file includes the necessary configuration to exclude Next.js files from the Python function build, preventing the "A Serverless Function has exceeded the unzipped maximum size of 250 MB" error.

## Environment Variables

Set these in your Vercel project settings:

- `MODEL_NAME` - The name of the LLM model to use (default: "gpt-3.5-turbo")
- `TEMPERATURE` - Temperature parameter for generation (default: 0.7)