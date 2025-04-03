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

  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Agents</h2>
        <div className="flex flex-wrap gap-2">
          {data.agents_names.map((name, index) => (
            <div
              key={index}
              className="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-full shadow-md"
            >
              {name}
            </div>
          ))}
        </div>
      </div>

      <h2 className="text-2xl font-bold text-gray-800 mb-4">Conversation Timeline</h2>

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
