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

  it('renders pipeline status with nested structure', () => {
    // API returns { status: 'ok', pipeline: { initialized: true, last_update: '...' } }
    const pipelineStatus = {
      status: 'ok',
      pipeline: {
        initialized: true,
        last_update: new Date().toISOString(),
      },
    };
    render(<AccountStatus pipelineStatus={pipelineStatus} />);

    expect(screen.getByText('Data Pipeline')).toBeInTheDocument();
    expect(screen.getByText('Just now')).toBeInTheDocument();
  });

  it('renders model status with loaded models', () => {
    // API returns { loaded: true, models: { '1H': {...}, '4H': {...}, 'D': {...} } }
    const modelStatus = {
      loaded: true,
      models: {
        '1H': { trained: true, val_accuracy: 0.67 },
        '4H': { trained: true, val_accuracy: 0.65 },
        'D': { trained: true, val_accuracy: 0.61 },
      },
    };
    render(<AccountStatus modelStatus={modelStatus} />);

    expect(screen.getByText('AI Models')).toBeInTheDocument();
    expect(screen.getByText('3 analyzers active')).toBeInTheDocument();
  });

  it('renders model accuracy when models are loaded', () => {
    const modelStatus = {
      loaded: true,
      models: {
        '1H': { trained: true, val_accuracy: 0.67 },
        '4H': { trained: true, val_accuracy: 0.65 },
        'D': { trained: true, val_accuracy: 0.61 },
      },
    };
    render(<AccountStatus modelStatus={modelStatus} />);

    expect(screen.getByText('Analysis Accuracy')).toBeInTheDocument();
    expect(screen.getByText('1H')).toBeInTheDocument();
    expect(screen.getByText('4H')).toBeInTheDocument();
    expect(screen.getByText('1D')).toBeInTheDocument();
    expect(screen.getByText('67.0%')).toBeInTheDocument();
  });

  it('displays not initialized when pipeline not initialized', () => {
    const pipelineStatus = {
      status: 'error',
      pipeline: {
        initialized: false,
      },
    };
    render(<AccountStatus pipelineStatus={pipelineStatus} />);

    expect(screen.getByText('Not initialized')).toBeInTheDocument();
  });

  it('displays not loaded when models not loaded', () => {
    const modelStatus = {
      loaded: false,
      models: {},
    };
    render(<AccountStatus modelStatus={modelStatus} />);

    expect(screen.getByText('Not loaded')).toBeInTheDocument();
  });
});
