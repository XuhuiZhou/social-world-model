# Social World Model UI

A beautiful web interface for visualizing agent interactions and conversations from the Social World Model.

## Features

- **Agent Presence Timeline**: Visualize when agents are present in the conversation
- **Conversation View**: See the conversation flow between agents
- **Timeline View**: Detailed view of each timestep with state, observations, and actions

## Getting Started

### Prerequisites

- Node.js 18.17 or later

### Installation

1. Clone the repository
2. Navigate to the UI directory:
   ```bash
   cd ui
   ```
3. Install dependencies:
   ```bash
   npm install
   ```

### Running the Development Server

```bash
npm run dev
```

This will start the development server on [http://localhost:3000](http://localhost:3000).

For allowing external access and specifying a port:

```bash
npm run dev -- -p 12000 -H 0.0.0.0
```

### Building for Production

```bash
npm run build
```

### Running in Production Mode

```bash
npm start
```

## Data Format

The UI expects data in the following JSON format:

```json
{
  "agents_names": [
    "Agent1",
    "Agent2",
    "Agent3"
  ],
  "socialized_context": [
    {
      "timestep": "1",
      "state": "Description of the current state",
      "observations": [
        "Agent1: observation",
        "Agent2: observation",
        "Agent3: observation"
      ],
      "actions": [
        "Agent1: action",
        "Agent2: action",
        "Agent3: action"
      ]
    }
  ]
}
```

## Technologies Used

- Next.js
- React
- TypeScript
- Tailwind CSS

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.
