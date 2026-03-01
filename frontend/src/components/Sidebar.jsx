export default function Sidebar({ activeTab, setActiveTab }) {
  return (
    <aside className="w-80 bg-slate-800/50 backdrop-blur-sm border-r border-slate-700 flex flex-col">
      <div className="p-6 border-b border-slate-700">
        <h2 className="text-2xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
          CineBot
        </h2>
      </div>

      <nav className="p-4 space-y-2">
        <button
          onClick={() => setActiveTab('chat')}
          className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
            activeTab === 'chat'
              ? 'bg-primary text-white shadow-lg shadow-primary/50'
              : 'text-slate-300 hover:bg-slate-700/50'
          }`}
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
          <span className="font-medium">Chat Recommender</span>
        </button>

        <button
          onClick={() => setActiveTab('poster')}
          className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
            activeTab === 'poster'
              ? 'bg-secondary text-white shadow-lg shadow-secondary/50'
              : 'text-slate-300 hover:bg-slate-700/50'
          }`}
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          <span className="font-medium">By Poster</span>
        </button>
      </nav>

      <div className="mt-auto p-6 border-t border-slate-700">
        <div className="bg-gradient-to-br from-primary/20 to-secondary/20 rounded-lg p-4 border border-primary/30">
          <h3 className="font-semibold text-sm mb-2">Open to Work</h3>
          <p className="text-xs text-slate-300 mb-2">
            Looking for roles in AI & Backend Engineering
          </p>
          <a
            href="https://www.janmajay.de"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-primary hover:text-secondary transition-colors underline"
          >
            janmajay.de
          </a>
        </div>
        <p className="text-center text-sm text-slate-400 mt-4">✍️ JANMAJAY KUMAR</p>
      </div>
    </aside>
  );
}
