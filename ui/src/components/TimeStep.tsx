"use client";

import React from 'react';
import { SocialContext } from '@/types';

interface TimeStepProps {
  data: SocialContext;
  agentNames: string[];
  isActive: boolean;
  onClick: () => void;
}

const TimeStep: React.FC<TimeStepProps> = ({ data, agentNames, isActive, onClick }) => {
  return (
    <div
      className={`border rounded-md p-5 mb-4 cursor-pointer transition-all ${
        isActive ? 'border-black bg-gray-50' : 'border-gray-200 hover:border-gray-300'
      }`}
      onClick={onClick}
    >
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-lg font-semibold text-gray-900">Timestep {data.timestep}</h3>
        {!isActive && (
          <span className="text-sm text-gray-500">Click to expand</span>
        )}
      </div>

      <p className="text-gray-700 mb-3">{data.state}</p>

      {isActive && (
        <div className="mt-5">
          <div className="mb-5">
            <h4 className="font-medium text-gray-900 mb-3">Observations:</h4>
            <ul className="space-y-2 pl-4">
              {agentNames.map(agent => (
                <li key={`obs-${agent}`} className="text-gray-600">
                  {data.observations[agent] !== 'none' ? (
                    <span>
                      <span className="font-medium">{agent}: </span>
                      {data.observations[agent] === '[SAME AS STATE]' ? data.state : data.observations[agent]}
                    </span>
                  ) : (
                    <span className="text-gray-400 italic">{agent} is not present</span>
                  )}
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="font-medium text-gray-900 mb-3">Actions:</h4>
            <ul className="space-y-2 pl-4">
              {agentNames.map(agent => (
                <li key={`act-${agent}`} className={`${data.actions[agent] === 'none' ? 'text-gray-400 italic' : 'text-gray-600'}`}>
                  <span className="font-medium">{agent}: </span>
                  {data.actions[agent]}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default TimeStep;
