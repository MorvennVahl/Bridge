import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: unknown) {
    // eslint-disable-next-line no-console
    console.error("ErrorBoundary caught:", error, info);
  }

  reset = () => this.setState({ error: null });

  render() {
    if (this.state.error) {
      return (
        <div className="mx-auto max-w-2xl p-8">
          <div className="rounded-xl border border-rose-200 bg-rose-50 p-6 shadow-sm">
            <h2 className="text-base font-semibold text-rose-800">Something broke on this page</h2>
            <p className="mt-1 text-sm text-rose-700">
              The rest of the app is still fine — pick another page in the sidebar. If you want to
              retry this one:
            </p>
            <div className="mt-4 flex gap-2">
              <button
                onClick={this.reset}
                className="rounded-md bg-slate-900 px-3 py-1.5 text-xs font-medium text-white"
              >
                Retry
              </button>
              <button
                onClick={() => window.location.reload()}
                className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700"
              >
                Full reload
              </button>
            </div>
            <pre className="mt-4 max-h-56 overflow-auto rounded bg-slate-900 p-3 text-[10px] leading-relaxed text-slate-100">
              {this.state.error.stack ?? this.state.error.message}
            </pre>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
