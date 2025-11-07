// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Home from './pages/Home';
import Login from './components/Auth/Login';
import Register from './components/Auth/Register';
import BuyerDashboard from './components/Buyer/BuyerDashboard';
import SellerDashboard from './components/Seller/SellerDashboard';
import Favourites from './components/Buyer/Favourites';
import SellerVisited from './components/Seller/SellerVisited';
import EMICalculator from './pages/EMICalculator';
import { isAuthenticated, getUserType } from './utils/auth';

const ProtectedRoute = ({ children, allowedUserType }) => {
  if (!isAuthenticated()) {
    return <Navigate to="/login" replace />;
  }

  const userType = getUserType();
  if (allowedUserType && userType !== allowedUserType) {
    return <Navigate to={`/${userType}-dashboard`} replace />;
  }

  return children;
};

const PublicRoute = ({ children }) => {
  if (isAuthenticated()) {
    const userType = getUserType();
    return <Navigate to={`/${userType}-dashboard`} replace />;
  }
  return children;
};

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        
        <Route
          path="/login"
          element={
            <PublicRoute>
              <Login />
            </PublicRoute>
          }
        />
        
        <Route
          path="/register"
          element={
            <PublicRoute>
              <Register />
            </PublicRoute>
          }
        />

        <Route
          path="/buyer-dashboard"
          element={
            <ProtectedRoute allowedUserType="buyer">
              <BuyerDashboard />
            </ProtectedRoute>
          }
        />
        
        <Route
          path="/favourites"
          element={
            <ProtectedRoute allowedUserType="buyer">
              <Favourites />
            </ProtectedRoute>
          }
        />
        
        <Route
          path="/emi-calculator"
          element={
            <ProtectedRoute allowedUserType="buyer">
              <EMICalculator />
            </ProtectedRoute>
          }
        />

        <Route
          path="/seller-dashboard"
          element={
            <ProtectedRoute allowedUserType="seller">
              <SellerDashboard />
            </ProtectedRoute>
          }
        />
        
        <Route
          path="/seller-visited"
          element={
            <ProtectedRoute allowedUserType="seller">
              <SellerVisited />
            </ProtectedRoute>
          }
        />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}

export default App;