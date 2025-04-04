from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from typing import Dict, List, Optional
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mock SocialWorldModel for demonstration
class MockSocialWorldModel:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.agents = {}
        self.task_specific_instructions = ""
        
    def reset_agents(self):
        self.agents = {}
        
    def set_task_specific_instructions(self, instructions):
        self.task_specific_instructions = instructions
        
    async def socialize_context(self, context, example_analysis="", feedback=""):
        # Mock implementation
        return {
            "agents_names": ["Alice", "Bob"],
            "socialized_context": [
                {
                    "timestep": 0,
                    "state": f"Alice and Bob are discussing: {context}",
                    "observations": {
                        "Alice": "You see Bob looking interested in the conversation.",
                        "Bob": "You see Alice explaining something with enthusiasm."
                    },
                    "actions": {
                        "Alice": "Alice explains her point of view.",
                        "Bob": "Bob nods in agreement."
                    }
                }
            ],
            "context_manual": f"Task specific instructions: {self.task_specific_instructions}"
        }
        
    async def initialize_simulation_from_socialized_context(self, context_obj):
        # Mock implementation
        self.agents = {name: {"memories": []} for name in context_obj["agents_names"]}
        
    async def reason_about_belief(self, question, agents, target_agent=None, answer_candidates=None):
        # Mock implementation
        reasoning = f"Thinking about how {target_agent or agents[0]} would answer: {question}"
        answer = f"{target_agent or agents[0]} would probably say: This is a mock response to the question."
        return reasoning, answer
        
    def get_simulation(self):
        # Mock implementation
        return {
            "agents": list(self.agents.keys()),
            "agent_memories": {agent: [] for agent in self.agents},
            "question": "",
            "reasoning": "",
            "answer": ""
        }

# Initialize a global MockSocialWorldModel instance
social_world_model = MockSocialWorldModel(
    model_name=os.environ.get("MODEL_NAME", "gpt-3.5-turbo"),
    temperature=float(os.environ.get("TEMPERATURE", "0.7"))
)

class SocializeContextRequest(BaseModel):
    context: str
    example_analysis: Optional[str] = ""
    feedback: Optional[str] = ""
    task_specific_instructions: Optional[str] = ""

class ReasonAboutBeliefRequest(BaseModel):
    question: str
    agents: List[str]
    target_agent: Optional[str] = None
    answer_candidates: Optional[List[str]] = None

@app.get("/api/ai")
async def root():
    return {"message": "Welcome to the Social World Model API (Mock Version)"}

@app.post("/api/ai/socialize-context")
async def socialize_context(request: SocializeContextRequest):
    try:
        # Set task-specific instructions if provided
        if request.task_specific_instructions:
            social_world_model.set_task_specific_instructions(request.task_specific_instructions)
        
        # Call the socialize_context method
        socialized_context = await social_world_model.socialize_context(
            context=request.context,
            example_analysis=request.example_analysis,
            feedback=request.feedback
        )
        
        # Return the result
        return socialized_context
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/initialize-simulation")
async def initialize_simulation(socialized_context: Dict = Body(...)):
    try:
        # Reset agents before initializing
        social_world_model.reset_agents()
        
        # Initialize the simulation
        await social_world_model.initialize_simulation_from_socialized_context(socialized_context)
        
        return {"message": "Simulation initialized successfully", "agents": list(social_world_model.agents.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/reason-about-belief")
async def reason_about_belief(request: ReasonAboutBeliefRequest):
    try:
        reasoning, answer = await social_world_model.reason_about_belief(
            question=request.question,
            agents=request.agents,
            target_agent=request.target_agent,
            answer_candidates=request.answer_candidates
        )
        
        return {
            "reasoning": reasoning,
            "answer": answer,
            "question": request.question,
            "target_agent": request.target_agent
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai/get-simulation")
async def get_simulation():
    try:
        simulation = social_world_model.get_simulation()
        return simulation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))