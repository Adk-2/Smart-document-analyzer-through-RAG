import ErrorBoundary from "./components/common/ErrorBoundary";
import Dashboard from "./pages/Dashboard";
import "./styles/global.css";

export default function App() {
  return (
    <ErrorBoundary label="Dashboard">
      <Dashboard />
    </ErrorBoundary>
  );
}
