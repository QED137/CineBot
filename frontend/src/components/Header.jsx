export default function Header() {
  return (
    <header className="bg-slate-800/30 backdrop-blur-sm border-b border-slate-700 px-6 py-4">
      <div className="overflow-hidden mb-3">
        <p className="text-xs text-yellow-400 animate-pulse">
          ⚠️ This app runs on Neo4j Free Tier and public APIs — Performance may vary!
        </p>
      </div>
      
      <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-2">
        AI Movie Recommender — Built with Flask, Neo4j & Multimodal RAG
      </h1>
      
      <p className="text-sm text-slate-400">
        End-to-end full-stack AI project using Graph DB, OpenAI, and CLIP embeddings
      </p>
    </header>
  );
}
