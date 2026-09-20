export default function Header() {
  return (
    <header className="header">
      <div className="workspace-title">
        <p className="eyebrow">RAG Analyzer</p>
        <h1>Research workspace</h1>
      </div>
      <div className="navbar-search" role="search">
        <input type="search" placeholder="Search sources or answers" aria-label="Search workspace" />
      </div>
      <nav className="navbar-actions" aria-label="Workspace actions">
        <button type="button" className="navbar-button">History</button>
        <button type="button" className="navbar-button navbar-button-primary">Export</button>
      </nav>
    </header>
  );
}
