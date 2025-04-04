"use client";

import React, { useState } from 'react';
import { SocialData } from '@/types';

interface ActionViewProps {
  data: SocialData;
}

interface Action {
  agent: string;
  content: string;
  timestep: string;
  observation?: string;
  state: string;
}

const ActionView: React.FC<ActionViewProps> = ({ data }) => {
  const { agents_names, socialized_context } = data;
  const [selectedAgents, setSelectedAgents] = useState<string[]>(agents_names);
  const [showState, setShowState] = useState(true);
  const [showCorresponding, setShowCorresponding] = useState(true);
  const [showAll, setShowAll] = useState(false);

  // Extract all actions and observations from the socialized context
  const actions: Action[] = [];
  socialized_context.forEach(context => {
    agents_names.forEach(agent => {
      const action = context.actions[agent];
      if (action !== 'none') {
        // Get the corresponding observation for this agent
        const observation = context.observations[agent];
        actions.push({
          agent,
          content: action,
          timestep: context.timestep,
          observation: observation === '[SAME AS STATE]' ? context.state : observation,
          state: context.state
        });
      }
    });
  });

  const toggleAgent = (agent: string) => {
    if (selectedAgents.includes(agent)) {
      setSelectedAgents(selectedAgents.filter(a => a !== agent));
    } else {
      setSelectedAgents([...selectedAgents, agent]);
    }
  };

  // Generate colors for agents
  const getAgentColorClass = (agent: string, type: 'bg' | 'text' | 'light') => {
    const index = agents_names.indexOf(agent);
    const colors = [
      { bg: 'bg-blue-600', text: 'text-blue-600', light: 'bg-blue-50' },
      { bg: 'bg-emerald-600', text: 'text-emerald-600', light: 'bg-emerald-50' },
      { bg: 'bg-violet-600', text: 'text-violet-600', light: 'bg-violet-50' },
      { bg: 'bg-amber-600', text: 'text-amber-600', light: 'bg-amber-50' },
      { bg: 'bg-rose-600', text: 'text-rose-600', light: 'bg-rose-50' },
      { bg: 'bg-cyan-600', text: 'text-cyan-600', light: 'bg-cyan-50' }
    ];
    return colors[index % colors.length][type];
  };

  const filteredActions = actions.filter(action => selectedAgents.includes(action.agent));

  return (
    <div className="w-full max-w-5xl mx-auto mb-12">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
        <h2 className="text-2xl font-bold text-gray-900 tracking-tight">Actions & Observations</h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setShowState(!showState)}
            className={`px-4 py-2 rounded-md font-medium transition-colors text-sm ${
              showState
                ? 'bg-black text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Show State
          </button>
          <button
            onClick={() => setShowCorresponding(!showCorresponding)}
            className={`px-4 py-2 rounded-md font-medium transition-colors text-sm ${
              showCorresponding
                ? 'bg-black text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Show Corresponding
          </button>
          <button
            onClick={() => setShowAll(!showAll)}
            className={`px-4 py-2 rounded-md font-medium transition-colors text-sm ${
              showAll
                ? 'bg-black text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Show All
          </button>
        </div>
      </div>

      <div className="mb-6 flex flex-wrap gap-2">
        {agents_names.map(agent => {
          const isSelected = selectedAgents.includes(agent);
          return (
            <button
              key={agent}
              onClick={() => toggleAgent(agent)}
              className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                isSelected
                  ? `${getAgentColorClass(agent, 'bg')} text-white`
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {agent}
            </button>
          );
        })}
      </div>

      <div className="space-y-6">
        {filteredActions.length > 0 ? (
          filteredActions.map((action, index) => (
            <div key={index} className="flex items-start">
              <div className={`flex-shrink-0 w-10 h-10 rounded-md ${getAgentColorClass(action.agent, 'bg')} flex items-center justify-center text-white font-bold`}>
                {action.agent.charAt(0)}
              </div>
              <div className="ml-3 flex-grow">
                <div className="flex items-baseline">
                  <span className={`font-medium ${getAgentColorClass(action.agent, 'text')}`}>{action.agent}</span>
                  <span className="ml-2 text-xs text-gray-500">Timestep {action.timestep}</span>
                </div>
                <div className="mt-2 space-y-3">
                  {showState && (
                    <div className="text-gray-700 bg-gray-50 p-3 rounded-md border border-gray-200">
                      <span className="font-medium text-gray-600">State: </span>
                      {action.state}
                    </div>
                  )}
                  {showAll ? (
                    // Show all observations for all agents at this timestep
                    Object.entries(socialized_context
                      .find(ctx => ctx.timestep === action.timestep)
                      ?.observations || {})
                      .map(([obsAgent, obsContent]) => {
                        if (obsContent === 'none') return null;
                        return (
                          <div key={obsAgent} className={`text-gray-700 ${getAgentColorClass(obsAgent, 'light')} p-3 rounded-md border border-gray-200`}>
                            <span className="font-medium text-gray-600">{obsAgent}'s Observation: </span>
                            {obsContent === '[SAME AS STATE]' ? action.state : obsContent}
                          </div>
                        );
                      })
                  ) : showCorresponding && (
                    // Show only the corresponding agent's observation
                    action.observation && action.observation !== 'none' && (
                      <div className={`text-gray-700 ${getAgentColorClass(action.agent, 'light')} p-3 rounded-md border border-gray-200`}>
                        <span className="font-medium text-gray-600">Observation: </span>
                        {action.observation}
                      </div>
                    )
                  )}
                  <div className="text-gray-700 bg-gray-50 p-3 rounded-md border border-gray-200">
                    <span className="font-medium text-gray-600">Action: </span>
                    {action.content}
                  </div>
                </div>
              </div>
            </div>
          ))
        ) : (
          <p className="text-gray-500 italic">No actions to display. Please select at least one agent.</p>
        )}
      </div>
    </div>
  );
};

export default ActionView;
