// src/components/shared/ChatBot.js
import React, { useState } from "react";
import { Send, Bot } from "lucide-react";
import axios from "axios";
import "./ChatBot.css";

const ChatBot = ({ onClose }) => {
  const [messages, setMessages] = useState([
    {
      type: "bot",
      text: "Hi! I'm your EstateAI assistant. How can I help you today?",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = input.trim();
    setMessages((prev) => [...prev, { type: "user", text: userMessage }]);
    setInput("");
    setLoading(true);

    try {
      const response = await axios.post("http://127.0.0.1:8000/chatbot", {
        message: userMessage,
      });

      setMessages((prev) => [
        ...prev,
        { type: "bot", text: response.data.response },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          type: "bot",
          text: "Sorry, I'm having trouble connecting. Please try again later.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chatbot-widget">
      <div className="chatbot-header">
        <div className="chatbot-header-content">
          <Bot size={24} />
          <div>
            <div className="chatbot-title">EstateAI Assistant</div>
            <div className="chatbot-status">Online</div>
          </div>
        </div>
      </div>

      <div className="chatbot-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`chat-message ${msg.type}`}>
            {msg.type === "bot" && (
              <div className="message-avatar">
                <Bot size={16} />
              </div>
            )}
            <div className="message-bubble">{msg.text}</div>
          </div>
        ))}
        {loading && (
          <div className="chat-message bot">
            <div className="message-avatar">
              <Bot size={16} />
            </div>
            <div className="message-bubble typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
      </div>

      <div className="chatbot-input">
        <input
          type="text"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
        />
        <button onClick={handleSend} disabled={!input.trim() || loading}>
          <Send size={20} />
        </button>
      </div>
    </div>
  );
};

export default ChatBot;
