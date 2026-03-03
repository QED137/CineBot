export default function EmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="text-center max-w-md">
        <div className="mb-6 inline-flex items-center justify-center w-24 h-24 bg-gradient-to-br from-primary/20 to-purple-500/20 rounded-full">
          <svg className="w-12 h-12 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
          </svg>
        </div>
        
        <h2 className="text-2xl font-bold text-white mb-3">
          Welcome to CineBot!
        </h2>
        
        <p className="text-slate-400 mb-6">
          Get personalized movie recommendations powered by AI. Ask for movies by genre, mood, or upload a poster to find similar films.
        </p>
        
        <div className="space-y-3 text-left bg-slate-800/30 rounded-lg p-4 border border-slate-700">
          <p className="text-sm text-slate-300 font-medium mb-2">Try asking:</p>
          <div className="space-y-2">
            <div className="flex items-start gap-2">
              <span className="text-primary mt-0.5">•</span>
              <span className="text-sm text-slate-400">"Show me sci-fi movies like Inception"</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-primary mt-0.5">•</span>
              <span className="text-sm text-slate-400">"I want a feel-good comedy"</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-primary mt-0.5">•</span>
              <span className="text-sm text-slate-400">"Recommend thriller movies from the 90s"</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-primary mt-0.5">•</span>
              <span className="text-sm text-slate-400">Upload a movie poster for visual similarity search</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
