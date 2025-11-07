// src/components/Auth/Register.js
import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { register } from "../../services/api";
import { setAuthData } from "../../utils/auth";
import { Home, ArrowLeft } from "lucide-react";
import "./Auth.css";

const Register = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    confirmPassword: "",
    full_name: "",
    user_type: "buyer",
    phone: "",
    whatsapp: "",
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

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    if (formData.password.length < 6) {
      setError("Password must be at least 6 characters");
      return;
    }

    setLoading(true);

    try {
      const { confirmPassword, ...registrationData } = formData;
      const response = await register(registrationData);
      const { token, user } = response.data;

      setAuthData(token, user);

      if (user.user_type === "buyer") {
        navigate("/buyer-dashboard");
      } else if (user.user_type === "seller") {
        navigate("/seller-dashboard");
      }
    } catch (err) {
      setError(
        err.response?.data?.detail || "Registration failed. Please try again."
      );
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
            <h2>Create Account</h2>
            <p>Join EstateAI to discover your dream property</p>
          </div>

          {error && <div className="error-message">{error}</div>}

          <form onSubmit={handleSubmit} className="auth-form">
            <div className="form-group">
              <label>Full Name</label>
              <input
                type="text"
                name="full_name"
                value={formData.full_name}
                onChange={handleChange}
                required
                placeholder="John Doe"
              />
            </div>

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
              <label>Phone Number</label>
              <input
                type="tel"
                name="phone"
                value={formData.phone}
                onChange={handleChange}
                placeholder="+91 1234567890"
              />
            </div>

            <div className="form-group">
              <label>WhatsApp Number (Optional)</label>
              <input
                type="tel"
                name="whatsapp"
                value={formData.whatsapp}
                onChange={handleChange}
                placeholder="+91 1234567890"
              />
            </div>

            <div className="form-group">
              <label>I want to</label>
              <select
                name="user_type"
                value={formData.user_type}
                onChange={handleChange}
                required
              >
                <option value="buyer">Buy Property</option>
                <option value="seller">Sell Property</option>
              </select>
            </div>

            <div className="form-group">
              <label>Password</label>
              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                required
                placeholder="At least 6 characters"
                minLength="6"
              />
            </div>

            <div className="form-group">
              <label>Confirm Password</label>
              <input
                type="password"
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleChange}
                required
                placeholder="Re-enter your password"
              />
            </div>

            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? "Creating Account..." : "Register"}
            </button>
          </form>

          <p className="auth-link">
            Already have an account? <Link to="/login">Login here</Link>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Register;
