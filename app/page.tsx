'use client';

import { useState, useEffect } from 'react';

export default function Home() {
  const [pythonMessage, setPythonMessage] = useState('');
  const [nextMessage, setNextMessage] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch from Python API
    fetch('/api/ai')
      .then(res => res.json())
      .then(data => {
        setPythonMessage(data.message);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error fetching from Python API:', err);
        setPythonMessage('Error fetching from Python API');
        setLoading(false);
      });

    // Fetch from Next.js API
    fetch('/api')
      .then(res => res.json())
      .then(data => {
        setNextMessage(data.message);
      })
      .catch(err => {
        console.error('Error fetching from Next.js API:', err);
        setNextMessage('Error fetching from Next.js API');
      });
  }, []);

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm">
        <h1 className="text-4xl font-bold mb-8">Social World Model</h1>
        
        <div className="mb-8 p-4 border rounded-lg">
          <h2 className="text-2xl font-semibold mb-4">Python API Response:</h2>
          {loading ? (
            <p>Loading...</p>
          ) : (
            <p className="text-lg">{pythonMessage}</p>
          )}
        </div>

        <div className="mb-8 p-4 border rounded-lg">
          <h2 className="text-2xl font-semibold mb-4">Next.js API Response:</h2>
          <p className="text-lg">{nextMessage}</p>
        </div>

        <div className="mb-8 p-4 border rounded-lg">
          <h2 className="text-2xl font-semibold mb-4">Try the Social World Model API:</h2>
          <p>
            <a href="/api/ai/socialize-context" className="text-blue-500 hover:underline">
              Socialize Context
            </a>
          </p>
          <p>
            <a href="/api/ai/get-simulation" className="text-blue-500 hover:underline">
              Get Simulation
            </a>
          </p>
        </div>
      </div>
    </main>
  );
}