import { useState } from 'react';
import { SocialData } from '@/types';

interface JsonInputModalProps {
  isOpen: boolean;
  onClose: () => void;
  onDataSubmit: (data: SocialData) => void;
}

export default function JsonInputModal({ isOpen, onClose, onDataSubmit }: JsonInputModalProps) {
  const [jsonInput, setJsonInput] = useState('');
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setJsonInput(e.target.value);
    setError(null);
  };

  const handleSubmit = () => {
    try {
      const parsedData = JSON.parse(jsonInput) as SocialData;

      // Basic validation
      if (!parsedData.agents_names || !Array.isArray(parsedData.agents_names)) {
        throw new Error('Invalid data: agents_names must be an array');
      }

      if (!parsedData.socialized_context || !Array.isArray(parsedData.socialized_context)) {
        throw new Error('Invalid data: socialized_context must be an array');
      }

      // Validate each context item
      parsedData.socialized_context.forEach((context, index) => {
        if (!context.timestep || !context.state) {
          throw new Error(`Invalid context at index ${index}: missing timestep or state`);
        }
        
        // Check if observations is an object
        if (!(typeof context.observations === 'object' && context.observations !== null)) {
          throw new Error(`Invalid context at index ${index}: observations must be an object`);
        }
        
        // Check if actions is an object
        if (!(typeof context.actions === 'object' && context.actions !== null)) {
          throw new Error(`Invalid context at index ${index}: actions must be an object`);
        }

        // Validate that all agents have entries in observations and actions
        parsedData.agents_names.forEach(agent => {
          if (!(agent in context.observations)) {
            throw new Error(`Invalid context at index ${index}: missing observation for agent ${agent}`);
          }
          if (!(agent in context.actions)) {
            throw new Error(`Invalid context at index ${index}: missing action for agent ${agent}`);
          }
        });
      });

      onDataSubmit(parsedData);
      setJsonInput('');
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Invalid JSON format');
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-md p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <h2 className="text-2xl font-bold mb-5 text-gray-900 tracking-tight">Paste JSON Data</h2>

        <div className="mb-5">
          <label htmlFor="json-input" className="block text-sm font-medium text-gray-700 mb-2">
            Enter JSON data in the format:
          </label>
          <pre className="bg-gray-900 p-4 rounded-md text-sm mb-4 overflow-x-auto font-mono leading-relaxed text-gray-100">
            {`{
  "agents_names": ["agent1", "agent2"],
  "socialized_context": [
    {
      "timestep": "t1",
      "state": "state description",
      "observations": {"agent1": "obs1", "agent2": "obs2"},
      "actions": {"agent1": "action1", "agent2": "action2"}
    }
  ]
}`}
          </pre>
          <textarea
            id="json-input"
            className="w-full h-64 p-4 border border-gray-300 rounded-md font-mono text-base leading-relaxed bg-gray-900 text-gray-100"
            value={jsonInput}
            onChange={handleInputChange}
            placeholder="Paste your JSON data here..."
            style={{ fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace' }}
          />
        </div>

        {error && (
          <div className="mb-5 p-3 bg-red-50 text-red-700 rounded-md font-medium border border-red-200">
            {error}
          </div>
        )}

        <div className="flex justify-end space-x-3">
          <button
            className="px-4 py-2 bg-gray-100 text-gray-800 rounded-md hover:bg-gray-200 font-medium text-sm"
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            className="px-4 py-2 bg-black text-white rounded-md hover:bg-gray-800 font-medium text-sm"
            onClick={handleSubmit}
          >
            Load Data
          </button>
        </div>
      </div>
    </div>
  );
}
