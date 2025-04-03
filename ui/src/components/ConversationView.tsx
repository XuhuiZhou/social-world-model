"use client";

import React, { useState } from 'react';
import { SocialData } from '@/types';

interface ConversationViewProps {
  data: SocialData;
}

interface Message {
  agent: string;
  content: string;
  timestep: string;
}

const ConversationView: React.FC<ConversationViewProps> = ({ data }) => {
  const { agents_names, socialized_context } = data;
  const [selectedAgents, setSelectedAgents] = useState<string[]>(agents_names);

  // Extract messages from actions
  const messages: Message[] = [];
  socialized_context.forEach(context => {
    context.actions.forEach((action, index) => {
      const agent = agents_names[index];
      if (!action.includes('none') && action.includes('says')) {
        const content = action.replace(`${agent}: says`, '').trim();
        messages.push({
          agent,
          content,
          timestep: context.timestep
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
  const getAgentColorClass = (agent: string, type: 'bg' | 'text') => {
    const index = agents_names.indexOf(agent);
    const colors = [
      { bg: 'bg-blue-500', text: 'text-blue-600' },
      { bg: 'bg-green-500', text: 'text-green-600' },
      { bg: 'bg-purple-500', text: 'text-purple-600' },
      { bg: 'bg-yellow-500', text: 'text-yellow-700' },
      { bg: 'bg-red-500', text: 'text-red-600' },
      { bg: 'bg-indigo-500', text: 'text-indigo-600' }
    ];
    return colors[index % colors.length][type];
  };

  const filteredMessages = messages.filter(msg => selectedAgents.includes(msg.agent));

  return (
    <div className="w-full max-w-4xl mx-auto mb-12">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Conversation</h2>
      
      <div className="mb-6 flex flex-wrap gap-2">
        {agents_names.map(agent => {
          const isSelected = selectedAgents.includes(agent);
          return (
            <button
              key={agent}
              onClick={() => toggleAgent(agent)}
              className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                isSelected 
                  ? `${getAgentColorClass(agent, 'bg')} text-white` 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {agent}
            </button>
          );
        })}
      </div>
      
      <div className="space-y-4">
        {filteredMessages.length > 0 ? (
          filteredMessages.map((msg, index) => (
            <div key={index} className="flex items-start">
              <div className={`flex-shrink-0 w-10 h-10 rounded-full ${getAgentColorClass(msg.agent, 'bg')} flex items-center justify-center text-white font-bold`}>
                {msg.agent.charAt(0)}
              </div>
              <div className="ml-3">
                <div className="flex items-baseline">
                  <span className={`font-medium ${getAgentColorClass(msg.agent, 'text')}`}>{msg.agent}</span>
                  <span className="ml-2 text-xs text-gray-500">Timestep {msg.timestep}</span>
                </div>
                <div className="mt-1 text-gray-700 bg-gray-100 p-3 rounded-lg">
                  {msg.content}
                </div>
              </div>
            </div>
          ))
        ) : (
          <p className="text-gray-500 italic">No messages to display. Please select at least one agent.</p>
        )}
      </div>
    </div>
  );
};

export default ConversationView;