import React from 'react';

type ChartType = 'bar' | 'line' | 'area' | 'hybrid';

interface VisualizationSwitcherProps {
  currentType: ChartType;
  onChange: (type: ChartType) => void;
}

export const VisualizationSwitcher: React.FC<VisualizationSwitcherProps> = ({
  currentType,
  onChange,
}) => {
  return (
    <div className="visualization-switcher">
      <button 
        className={`btn ${currentType === 'bar' ? 'active' : ''}`}
        onClick={() => onChange('bar')}
      >
        Bar
      </button>
      <button 
        className={`btn ${currentType === 'line' ? 'active' : ''}`}
        onClick={() => onChange('line')}
      >
        Line
      </button>
      <button 
        className={`btn ${currentType === 'area' ? 'active' : ''}`}
        onClick={() => onChange('area')}
      >
        Area
      </button>
      <button 
        className={`btn ${currentType === 'hybrid' ? 'active' : ''}`}
        onClick={() => onChange('hybrid')}
      >
        Hybrid
      </button>
    </div>
  );
};
