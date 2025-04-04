# Social World Model - Vercel Python API

This branch adds Vercel Functions support to the Social World Model project, allowing you to deploy the SocialWorldModel class as a serverless API.

## Structure

- `/api/index.py` - The main FastAPI application that exposes the SocialWorldModel functionality
- `/public/index.html` - A simple web interface for testing the API
- `/vercel.json` - Configuration for Vercel deployment
- `/requirements.txt` - Dependencies for the Vercel Python runtime

## API Endpoints

- `GET /` - Welcome message
- `POST /socialize-context` - Analyzes and socializes a context for simulation
- `POST /initialize-simulation` - Initializes a simulation from a socialized context
- `POST /reason-about-belief` - Reasons about an agent's belief in response to a question
- `GET /get-simulation` - Retrieves the current state of the simulation

## Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the API locally:
   ```
   cd api
   uvicorn index:app --host 0.0.0.0 --port 8000 --reload
   ```

3. Access the web interface at http://localhost:8000

## Deployment to Vercel

1. Install Vercel CLI:
   ```
   npm install -g vercel
   ```

2. Deploy to Vercel:
   ```
   vercel
   ```

## Environment Variables

Set these in your Vercel project settings:

- `MODEL_NAME` - The name of the LLM model to use (default: "gpt-3.5-turbo")
- `TEMPERATURE` - Temperature parameter for generation (default: 0.7)

## Notes

- The API uses FastAPI with the Vercel Python runtime
- The SocialWorldModel class is imported directly from the project
- CORS is enabled for all origins to allow API access from any frontend