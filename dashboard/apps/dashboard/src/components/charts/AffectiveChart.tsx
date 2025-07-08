import React, { useContext, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, AreaChart, Area } from 'recharts';
import AffectiveContext from '../../contexts/AffectiveContext';
import { VisualizationSwitcher } from '../VisualizationSwitcher';

interface AffectiveChartProps {
  data: Array<{
    name: string;
    value: number;
  }>;
}

type ChartType = 'bar' | 'line' | 'area' | 'hybrid';

export const AffectiveChart = () => {
  const { data, loading, error } = useContext(AffectiveContext);
  const [chartType, setChartType] = useState<ChartType>('bar');

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <p>Loading affective data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-500 p-4">
        Error: {error}
      </div>
    );
  }

  if (!data.length) {
    return (
      <div className="p-4 text-gray-500">
        No affective data available
      </div>
    );
  }

  return (
    <div className="affective-chart">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">Affective State</h3>
        <VisualizationSwitcher 
          currentType={chartType} 
          onChange={setChartType} 
        />
      </div>

      {renderChart()}
    </div>
  );

  function renderChart() {
    switch (chartType) {
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="value" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        );
      case 'line':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="value" stroke="#8884d8" />
            </LineChart>
          </ResponsiveContainer>
        );
      case 'area':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Area type="monotone" dataKey="value" fill="#8884d8" />
            </AreaChart>
          </ResponsiveContainer>
        );
      default:
        return <div>Chart type not supported</div>;
    }
  }
};
