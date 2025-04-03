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
    const colors = [
      { bg: 'bg-blue-600', border: 'border-blue-700' },
      { bg: 'bg-emerald-600', border: 'border-emerald-700' },
      { bg: 'bg-violet-600', border: 'border-violet-700' },
      { bg: 'bg-amber-600', border: 'border-amber-700' },
      { bg: 'bg-rose-600', border: 'border-rose-700' },
      { bg: 'bg-cyan-600', border: 'border-cyan-700' }
    ];
    return colors[agentIndex % colors.length];
  };

  return (
    <div className="w-full max-w-5xl mx-auto mb-12">
      <h2 className="text-2xl font-bold text-gray-900 mb-6 tracking-tight">Agent Presence Timeline</h2>

      <div className="overflow-x-auto">
        <div className="min-w-max">
          {/* Timeline header */}
          <div className="flex mb-4">
            <div className="w-32 flex-shrink-0"></div>
            <div className="flex-1 grid grid-cols-14 gap-1">
              {socialized_context.map((context) => (
                <div key={context.timestep} className="h-8 flex items-center justify-center">
                  <span className="text-sm font-medium text-gray-600">{context.timestep}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Agent rows */}
          {agents_names.map((name, agentIndex) => {
            const colorClasses = getAgentColorClass(agentIndex);
            return (
              <div key={name} className="flex mb-3 items-center">
                <div className="w-32 flex-shrink-0">
                  <span className={`font-medium ${colorClasses.bg.replace('bg-', 'text-')}`}>{name}</span>
                </div>
                <div className="flex-1 grid grid-cols-14 gap-1">
                  {socialized_context.map((context) => {
                    const present = isAgentPresent(agentIndex, context.timestep);
                    return (
                      <div
                        key={context.timestep}
                        className={`h-8 rounded-md border transition-all ${
                          present
                            ? `${colorClasses.bg} ${colorClasses.border} opacity-100`
                            : 'border-gray-200 bg-gray-50 opacity-50'
                        }`}
                        title={`${name} - Timestep ${context.timestep}`}
                      />
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default AgentPresence;
