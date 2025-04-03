import Timeline from '@/components/Timeline';
import AgentPresence from '@/components/AgentPresence';
import ConversationView from '@/components/ConversationView';
import socialData from '@/data/socialData.json';
import { SocialData } from '@/types';

export default function Home() {
  const data = socialData as SocialData;

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      <header className="bg-gradient-to-r from-blue-600 to-purple-600 text-white py-8 px-4 shadow-lg">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl md:text-4xl font-bold mb-2">Social World Model</h1>
          <p className="text-blue-100">Visualizing agent interactions and conversations</p>
        </div>
      </header>

      <main className="py-8 px-4">
        <div className="max-w-4xl mx-auto space-y-12">
          <section>
            <AgentPresence data={data} />
          </section>

          <section>
            <ConversationView data={data} />
          </section>

          <section>
            <Timeline data={data} />
          </section>
        </div>
      </main>

      <footer className="bg-gray-800 text-white py-6 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <p>Social World Model Visualization - {new Date().getFullYear()}</p>
        </div>
      </footer>
    </div>
  );
}
