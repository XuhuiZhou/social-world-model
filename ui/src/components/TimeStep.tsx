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
      className={`border rounded-lg p-4 mb-4 cursor-pointer transition-all ${
        isActive ? 'border-blue-500 bg-blue-50 shadow-md' : 'border-gray-200 hover:border-blue-300'
      }`}
      onClick={onClick}
    >
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-lg font-semibold">Timestep {data.timestep}</h3>
        {!isActive && (
          <span className="text-sm text-gray-500">Click to expand</span>
        )}
      </div>
      
      <p className="text-gray-700 mb-2">{data.state}</p>
      
      {isActive && (
        <div className="mt-4">
          <div className="mb-4">
            <h4 className="font-medium text-gray-800 mb-2">Observations:</h4>
            <ul className="space-y-1 pl-4">
              {data.observations.map((observation, index) => (
                <li key={`obs-${index}`} className="text-gray-600">
                  {observation !== `${agentNames[index]}: none` ? observation : (
                    <span className="text-gray-400 italic">{agentNames[index]} is not present</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-800 mb-2">Actions:</h4>
            <ul className="space-y-1 pl-4">
              {data.actions.map((action, index) => (
                <li key={`act-${index}`} className={`${action.includes('none') ? 'text-gray-400 italic' : 'text-gray-600'}`}>
                  {action}
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