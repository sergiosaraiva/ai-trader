import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AccountStatus } from './AccountStatus';

describe('AccountStatus', () => {
  it('renders loading state', () => {
    render(<AccountStatus loading={true} />);
    const skeleton = document.querySelector('.animate-pulse');
    expect(skeleton).toBeInTheDocument();
  });

  it('renders error state', () => {
    render(<AccountStatus error="Connection failed" />);
    expect(screen.getByText('Connection failed')).toBeInTheDocument();
  });

  it('renders system status header', () => {
    render(<AccountStatus />);
    expect(screen.getByText('System Status')).toBeInTheDocument();
  });

  it('renders pipeline status', () => {
    const pipelineStatus = {
      status: 'healthy',
      last_run: new Date().toISOString(),
    };
    render(<AccountStatus pipelineStatus={pipelineStatus} />);

    expect(screen.getByText('Data Pipeline')).toBeInTheDocument();
    expect(screen.getByText('Just now')).toBeInTheDocument();
  });

  it('renders model status', () => {
    const modelStatus = {
      models_loaded: true,
    };
    render(<AccountStatus modelStatus={modelStatus} />);

    expect(screen.getByText('ML Models')).toBeInTheDocument();
    expect(screen.getByText('All models loaded')).toBeInTheDocument();
  });

  it('renders data quality when available', () => {
    const pipelineStatus = {
      status: 'healthy',
      last_run: new Date().toISOString(),
      data_quality: {
        '1H': { status: 'healthy', rows: 1000 },
        '4H': { status: 'healthy', rows: 250 },
        'D': { status: 'healthy', rows: 40 },
      },
    };
    render(<AccountStatus pipelineStatus={pipelineStatus} />);

    expect(screen.getByText('Data Quality')).toBeInTheDocument();
    expect(screen.getByText('1H')).toBeInTheDocument();
    expect(screen.getByText('4H')).toBeInTheDocument();
    expect(screen.getByText('D')).toBeInTheDocument();
  });

  it('displays not initialized when no last_run', () => {
    const pipelineStatus = {
      status: 'unknown',
    };
    render(<AccountStatus pipelineStatus={pipelineStatus} />);

    expect(screen.getByText('Not initialized')).toBeInTheDocument();
  });
});
