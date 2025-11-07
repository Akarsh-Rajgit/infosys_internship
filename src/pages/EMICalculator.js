// src/pages/EMICalculator.js
import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { getAuthData, clearAuthData } from "../utils/auth";
import {
  Home,
  LogOut,
  ArrowLeft,
  Calculator,
  TrendingDown,
} from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts";
import "./EMICalculator.css";

const banks = [
  { name: "SBI", rate: 8.5, processingFee: 0.5 },
  { name: "HDFC", rate: 8.75, processingFee: 0.5 },
  { name: "ICICI", rate: 8.65, processingFee: 0.75 },
  { name: "Axis Bank", rate: 8.9, processingFee: 1.0 },
  { name: "Kotak Mahindra", rate: 8.7, processingFee: 0.5 },
  { name: "Punjab National Bank", rate: 8.55, processingFee: 0.35 },
];

const EMICalculator = () => {
  const navigate = useNavigate();
  const { user } = getAuthData();
  const [loanAmount, setLoanAmount] = useState(5000000);
  const [tenure, setTenure] = useState(20);
  const [selectedBank, setSelectedBank] = useState(banks[0]);

  const calculateEMI = (principal, rate, tenure) => {
    const monthlyRate = rate / 12 / 100;
    const months = tenure * 12;
    const emi =
      (principal * monthlyRate * Math.pow(1 + monthlyRate, months)) /
      (Math.pow(1 + monthlyRate, months) - 1);
    return emi;
  };

  const emi = calculateEMI(loanAmount, selectedBank.rate, tenure);
  const totalPayment = emi * tenure * 12;
  const totalInterest = totalPayment - loanAmount;
  const processingFee = (loanAmount * selectedBank.processingFee) / 100;

  const chartData = [
    { name: "Principal", value: loanAmount, color: "#667eea" },
    { name: "Interest", value: totalInterest, color: "#764ba2" },
  ];

  const handleLogout = () => {
    clearAuthData();
    navigate("/login");
  };

  return (
    <div className="dashboard">
      <nav className="navbar">
        <div className="navbar-brand">
          <Home size={24} />
          <span>EstateAI - EMI Calculator</span>
        </div>
        <div className="navbar-actions">
          <Link to="/buyer-dashboard" className="nav-icon-btn">
            <ArrowLeft size={20} /> <font color="white">Back to dashboard</font>
          </Link>
          <div className="navbar-user">
            <span>{user?.full_name || user?.email}</span>
          </div>
          <button onClick={handleLogout} className="btn-logout">
            <LogOut size={18} /> Logout
          </button>
        </div>
      </nav>

      <div className="dashboard-content">
        <div className="page-header">
          <Calculator size={32} className="page-icon" />
          <div>
            <h1 className="dashboard-title">EMI Calculator</h1>
            <p className="page-subtitle">
              Calculate your home loan EMI from various banks
            </p>
          </div>
        </div>

        <div className="emi-calculator-container">
          <div className="calculator-inputs">
            <div className="input-section">
              <label>Loan Amount</label>
              <div className="input-with-label">
                <span className="currency">₹</span>
                <input
                  type="number"
                  value={loanAmount}
                  onChange={(e) => setLoanAmount(Number(e.target.value))}
                  min="100000"
                  step="100000"
                />
              </div>
              <input
                type="range"
                min="100000"
                max="50000000"
                step="100000"
                value={loanAmount}
                onChange={(e) => setLoanAmount(Number(e.target.value))}
                className="range-slider"
              />
              <div className="range-labels">
                <span>₹1L</span>
                <span>₹5Cr</span>
              </div>
            </div>

            <div className="input-section">
              <label>Loan Tenure</label>
              <div className="input-with-label">
                <input
                  type="number"
                  value={tenure}
                  onChange={(e) => setTenure(Number(e.target.value))}
                  min="1"
                  max="30"
                />
                <span className="unit">years</span>
              </div>
              <input
                type="range"
                min="1"
                max="30"
                value={tenure}
                onChange={(e) => setTenure(Number(e.target.value))}
                className="range-slider"
              />
              <div className="range-labels">
                <span>1 year</span>
                <span>30 years</span>
              </div>
            </div>

            <div className="input-section">
              <label>Select Bank</label>
              <div className="bank-options">
                {banks.map((bank) => (
                  <button
                    key={bank.name}
                    className={`bank-option ${
                      selectedBank.name === bank.name ? "active" : ""
                    }`}
                    onClick={() => setSelectedBank(bank)}
                  >
                    <div className="bank-name">{bank.name}</div>
                    <div className="bank-rate">{bank.rate}% p.a.</div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="calculator-results">
            <div className="result-card primary">
              <div className="result-label">Monthly EMI</div>
              <div className="result-value">
                ₹{emi.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
              </div>
            </div>

            <div className="results-grid">
              <div className="result-card">
                <div className="result-label">Principal Amount</div>
                <div className="result-value small">
                  ₹{(loanAmount / 100000).toFixed(2)}L
                </div>
              </div>
              <div className="result-card">
                <div className="result-label">Total Interest</div>
                <div className="result-value small">
                  ₹{(totalInterest / 100000).toFixed(2)}L
                </div>
              </div>
              <div className="result-card">
                <div className="result-label">Total Payment</div>
                <div className="result-value small">
                  ₹{(totalPayment / 100000).toFixed(2)}L
                </div>
              </div>
              <div className="result-card">
                <div className="result-label">Processing Fee</div>
                <div className="result-value small">
                  ₹
                  {processingFee
                    .toFixed(0)
                    .replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
                </div>
              </div>
            </div>

            <div className="chart-container">
              <h3>Loan Breakdown</h3>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) =>
                      `${name}: ${(percent * 100).toFixed(0)}%`
                    }
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(value) => `₹${(value / 100000).toFixed(2)}L`}
                  />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="bank-comparison">
              <h3>
                <TrendingDown size={20} /> Compare All Banks
              </h3>
              <div className="comparison-table">
                <table>
                  <thead>
                    <tr>
                      <th>Bank</th>
                      <th>Interest Rate</th>
                      <th>Monthly EMI</th>
                      <th>Total Interest</th>
                    </tr>
                  </thead>
                  <tbody>
                    {banks.map((bank) => {
                      const bankEMI = calculateEMI(
                        loanAmount,
                        bank.rate,
                        tenure
                      );
                      const bankTotal = bankEMI * tenure * 12;
                      const bankInterest = bankTotal - loanAmount;
                      return (
                        <tr
                          key={bank.name}
                          className={
                            selectedBank.name === bank.name ? "selected" : ""
                          }
                        >
                          <td>{bank.name}</td>
                          <td>{bank.rate}%</td>
                          <td>
                            ₹
                            {bankEMI
                              .toFixed(0)
                              .replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
                          </td>
                          <td>₹{(bankInterest / 100000).toFixed(2)}L</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EMICalculator;
