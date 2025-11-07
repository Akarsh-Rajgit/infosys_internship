// src/components/Auth/Login.js
import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { login } from "../../services/api";
import { setAuthData } from "../../utils/auth";
import { Home, ArrowLeft } from "lucide-react";
import "./Auth.css";

const Login = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const response = await login(formData);
      const { token, user } = response.data;

      setAuthData(token, user);

      if (user.user_type === "buyer") {
        navigate("/buyer-dashboard");
      } else if (user.user_type === "seller") {
        navigate("/seller-dashboard");
      }
    } catch (err) {
      setError(err.response?.data?.detail || "Login failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <Link to="/" className="back-home">
        <ArrowLeft size={20} />
        Back to Home
      </Link>

      <div className="auth-container">
        <div className="auth-card animated">
          <div className="auth-header">
            <Home size={40} className="auth-icon" />
            <h2>Welcome Back</h2>
            <p>Login to access your real estate dashboard</p>
          </div>

          {error && <div className="error-message">{error}</div>}

          <form onSubmit={handleSubmit} className="auth-form">
            <div className="form-group">
              <label>Email Address</label>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                required
                placeholder="your.email@example.com"
              />
            </div>

            <div className="form-group">
              <label>Password</label>
              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                required
                placeholder="Enter your password"
              />
            </div>

            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? "Logging in..." : "Login"}
            </button>
          </form>

          <p className="auth-link">
            Don't have an account? <Link to="/register">Register here</Link>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
