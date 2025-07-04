# ModelPulse Frontend

This is the frontend for ModelPulse, a real-time ML model monitoring platform that allows you to track and analyze model inference data.

## Features

- Real-time dashboard with key performance indicators
- Live logs table with filtering and sorting
- Visual analytics with charts and graphs
- Alert system for monitoring model performance
- Settings page for configuring alert thresholds

## Tech Stack

- React.js with TypeScript
- Ant Design for UI components
- Recharts for data visualization
- Axios for API requests
- React Router for navigation

## Getting Started

### Prerequisites

- Node.js 14+ and npm

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/modelpulse.git
cd modelpulse/frontend
```

2. Install dependencies:

```bash
npm install
```

3. Create a `.env` file in the frontend directory with the following content:

```
VITE_API_URL=http://localhost:8000
```

Replace the URL with your backend API URL if it's different.

### Running the Development Server

```bash
npm run dev
```

The application will be available at http://localhost:3000.

### Building for Production

```bash
npm run build
```

The build artifacts will be stored in the `dist/` directory.

## Project Structure

```
frontend/
├── public/             # Static assets
├── src/
│   ├── components/     # Reusable components
│   ├── pages/          # Page components
│   ├── services/       # API services
│   ├── types/          # TypeScript interfaces
│   ├── App.tsx         # Main application component
│   ├── main.tsx        # Entry point
│   └── index.css       # Global styles
├── .env                # Environment variables
├── package.json        # Dependencies and scripts
├── tsconfig.json       # TypeScript configuration
└── vite.config.ts      # Vite configuration
```

## Features

### Dashboard

The dashboard provides a quick overview of your model's performance with:

- Key performance indicators (average latency, confidence, etc.)
- Visual indicators for KPIs that exceed thresholds (yellow for warning, red for error)
- Recent inference logs
- Alert notifications for threshold breaches

### Logs Table

The logs table shows detailed information about model inferences:

- Model name, timestamp, input shape, latency, confidence, and output class
- Visual highlighting of values and rows that exceed thresholds
- Filtering by model name and output class
- Sorting by various columns
- Pagination for large datasets

### Analytics

The analytics page provides visual insights into your model's performance:

- Latency over time chart
- Confidence distribution histogram
- Model usage distribution pie chart
- Class distribution bar chart

### Settings

The settings page allows you to configure alert thresholds:

- Create, edit, and delete alert thresholds
- Set thresholds for specific models or all models
- Configure alerts for latency and confidence metrics
- Enable/disable alert thresholds

## Integration with Backend

The frontend communicates with the ModelPulse backend API to:

- Fetch inference logs and analytics data
- Manage alert thresholds
- Receive alerts when thresholds are breached

## License

MIT
