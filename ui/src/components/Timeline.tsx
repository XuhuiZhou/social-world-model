"use client";

import React, { useState } from 'react';
import { SocialData } from '@/types';
import TimeStep from './TimeStep';

interface TimelineProps {
  data: SocialData;
}

const Timeline: React.FC<TimelineProps> = ({ data }) => {
  const [activeStep, setActiveStep] = useState<string | null>(null);

  const handleStepClick = (timestep: string) => {
    setActiveStep(activeStep === timestep ? null : timestep);
  };

  // Generate colors for agents
  const getAgentColorClass = (index: number) => {
    const colors = [
      'bg-blue-600',
      'bg-emerald-600',
      'bg-violet-600',
      'bg-amber-600',
      'bg-rose-600',
      'bg-cyan-600'
    ];
    return colors[index % colors.length];
  };

  return (
    <div className="w-full max-w-5xl mx-auto">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-gray-900 tracking-tight mb-4">Agents</h2>
        <div className="flex flex-wrap gap-2">
          {data.agents_names.map((name, index) => (
            <div
              key={index}
              className={`px-4 py-2 ${getAgentColorClass(index)} text-white rounded-md text-sm font-medium`}
            >
              {name}
            </div>
          ))}
        </div>
      </div>

      <h2 className="text-2xl font-bold text-gray-900 tracking-tight mb-6">Conversation Timeline</h2>

      <div className="space-y-4">
        {data.socialized_context.map((context) => (
          <TimeStep
            key={context.timestep}
            data={context}
            agentNames={data.agents_names}
            isActive={activeStep === context.timestep}
            onClick={() => handleStepClick(context.timestep)}
          />
        ))}
      </div>
    </div>
  );
};

export default Timeline;
