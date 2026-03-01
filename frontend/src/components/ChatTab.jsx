import { useState, useRef, useEffect } from 'react';
import { chatAPI } from '../services/api';
import MovieCard from './MovieCard';

export default function ChatTab() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [placeholder, setPlaceholder] = useState('Ask for a movie or a follow-up question...');
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [inputValue]);

  const fetchSuggestion = async () => {
    try {
      const data = await chatAPI.getSuggestion();
      if (data.suggestion) {
        setPlaceholder(data.suggestion);
      }
    } catch (error) {
      console.error('Failed to fetch suggestion:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    setPlaceholder('');
    
    // Add user message to chat
    const newUserMessage = { role: 'user', content: userMessage };
    setMessages(prev => [...prev, newUserMessage]);
    setIsLoading(true);

    try {
      // Get chat history for API call
      const chatHistory = [...messages, newUserMessage];
      
      const data = await chatAPI.sendTextQuery(userMessage, chatHistory);
      
      // Add assistant response
      const assistantMessage = {
        role: 'assistant',
        content: data.llm_response_text,
        movies: data.context_movies || [],
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        role: 'assistant',
        content: `Oops! An error occurred: ${error.message}`,
        isError: true,
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setPlaceholder('Ask for a movie or a follow-up question...');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="text-6xl mb-4">🎬</div>
            <h2 className="text-2xl font-bold mb-2">Welcome to CineBot!</h2>
            <p className="text-slate-400 max-w-md">
              Ask me about movies, request recommendations, or upload a poster to find similar films.
            </p>
          </div>
        ) : (
          messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-3xl ${
                  message.role === 'user'
                    ? 'bg-primary text-white rounded-2xl rounded-tr-sm'
                    : 'bg-slate-800/50 text-slate-100 rounded-2xl rounded-tl-sm'
                } px-4 py-3 ${message.isError ? 'border border-red-500' : ''}`}
              >
                <p className="whitespace-pre-wrap">{message.content}</p>
                
                {/* Display movie cards if available */}
                {message.movies && message.movies.length > 0 && (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-4">
                    {message.movies.slice(0, 3).map((movie, idx) => (
                      <MovieCard key={idx} movie={movie} index={idx} />
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-slate-800/50 rounded-2xl rounded-tl-sm px-4 py-3">
              <div className="flex space-x-2">
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-slate-700 bg-slate-800/30 backdrop-blur-sm p-4">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              rows={1}
              className="w-full bg-slate-700/50 text-white rounded-xl px-4 py-3 pr-12 resize-none focus:outline-none focus:ring-2 focus:ring-primary max-h-32"
              disabled={isLoading}
            />
            <button
              type="button"
              onClick={fetchSuggestion}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-2 text-slate-400 hover:text-primary transition-colors"
              title="Get inspiration"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </button>
          </div>
          
          <button
            type="submit"
            disabled={isLoading || !inputValue.trim()}
            className="bg-primary hover:bg-primary/80 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded-xl px-6 py-3 transition-all flex items-center gap-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
