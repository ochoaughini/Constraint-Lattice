import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Button } from './Button';


describe('Button component', () => {
  it('renders provided children', () => {
    render(<Button>Click</Button>);
    expect(screen.getByText('Click')).toBeDefined();
  });

  it('applies custom className', () => {
    render(<Button className="custom">Text</Button>);
    const btn = screen.getByText('Text');
    expect(btn.className).toContain('custom');
  });

  it('triggers onClick handler', () => {
    const handler = vi.fn();
    render(<Button onClick={handler}>Go</Button>);
    screen.getByText('Go').click();
    expect(handler).toHaveBeenCalledTimes(1);
  });
});
