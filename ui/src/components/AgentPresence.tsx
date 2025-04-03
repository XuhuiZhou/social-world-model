"use client";

import React from 'react';
import { SocialData } from '@/types';

interface AgentPresenceProps {
  data: SocialData;
}

const AgentPresence: React.FC<AgentPresenceProps> = ({ data }) => {
  const { agents_names, socialized_context } = data;
  
  // Determine if an agent is present in a given timestep
  const isAgentPresent = (agentIndex: number, timestep: string) => {
    const context = socialized_context.find(c => c.timestep === timestep);
    if (!context) return false;
    
    return !context.observations[agentIndex].includes('none');
  };
  
  // Generate colors for agents
  const getAgentColorClass = (agentIndex: number) => {
    const agentColors = [
      'bg-blue-500',
      'bg-green-500',
      'bg-purple-500',
      'bg-yellow-500',
      'bg-red-500',
      'bg-indigo-500',
    ];
    return agentColors[agentIndex % agentColors.length];
  };
  
  return (
    <div className="w-full max-w-4xl mx-auto mb-12">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Agent Presence Timeline</h2>
      
      <div className="overflow-x-auto">
        <div className="min-w-max">
          <div className="flex mb-2">
            <div className="w-24 flex-shrink-0"></div>
            {socialized_context.map((context) => (
              <div key={context.timestep} className="w-10 text-center font-medium text-gray-700">
                {context.timestep}
              </div>
            ))}
          </div>
          
          {agents_names.map((name, agentIndex) => (
            <div key={name} className="flex items-center mb-3">
              <div className="w-24 flex-shrink-0 font-medium text-gray-800">{name}</div>
              {socialized_context.map((context) => (
                <div key={context.timestep} className="w-10 flex justify-center items-center">
                  <div 
                    className={`w-6 h-6 rounded-full ${
                      isAgentPresent(agentIndex, context.timestep) 
                        ? getAgentColorClass(agentIndex) 
                        : 'bg-gray-200'
                    }`}
                  ></div>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default AgentPresence;