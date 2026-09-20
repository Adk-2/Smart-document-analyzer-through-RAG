import { Component } from 'react';


export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error(`${this.props.label || 'Component'} crashed`, error, info);
  }

  handleRetry = () => {
    this.setState({ error: null });
  };

  render() {
    if (this.state.error) {
      return (
        <section className={this.props.className || 'panel'}>
          <div className="panel-heading">
            <h2>{this.props.label || 'Section'} unavailable</h2>
            <span>Render error</span>
          </div>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            This section failed to render.
          </span>
          <button
            type="button"
            className="navbar-button"
            onClick={this.handleRetry}
            style={{ marginTop: '0.75rem', width: 'fit-content' }}
          >
            Retry
          </button>
        </section>
      );
    }

    return this.props.children;
  }
}
