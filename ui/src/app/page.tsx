'use client';

import { useState } from 'react';
import Timeline from '@/components/Timeline';
import AgentPresence from '@/components/AgentPresence';
import ActionView from '@/components/ActionView';
import JsonInputModal from '@/components/JsonInputModal';
import socialData from '@/data/socialData.json';
import { SocialData } from '@/types';

export default function Home() {
  const [data, setData] = useState<SocialData>(socialData as SocialData);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleDataSubmit = (newData: SocialData) => {
    setData(newData);
  };

  return (
    <div className="min-h-screen bg-white">
      <header className="bg-black text-white py-10 px-6">
        <div className="max-w-5xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
          <div>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-3">Social World Model</h1>
            <p className="text-gray-400 text-lg">Visualizing agent interactions and states</p>
          </div>
          <button
            onClick={() => setIsModalOpen(true)}
            className="bg-white text-black px-5 py-2.5 rounded-md font-medium hover:bg-gray-100 transition-colors text-sm"
          >
            Load Custom Data
          </button>
        </div>
      </header>

      <main className="py-12 px-6">
        <div className="max-w-5xl mx-auto space-y-16">
          <section>
            <AgentPresence data={data} />
          </section>

          <section>
            <ActionView data={data} />
          </section>

          <section>
            <Timeline data={data} />
          </section>
        </div>
      </main>

      <footer className="bg-gray-50 text-gray-600 py-8 px-6 border-t border-gray-200">
        <div className="max-w-5xl mx-auto text-center">
          <p className="text-sm">Social World Model Visualization - {new Date().getFullYear()}</p>
        </div>
      </footer>

      <JsonInputModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onDataSubmit={handleDataSubmit}
      />
    </div>
  );
}
