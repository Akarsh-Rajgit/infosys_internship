// src/components/Buyer/Favourites.js
import React, { useState, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import {
  getFavourites,
  removeFavourite,
  predictBulk,
} from "../../services/api";
import { getAuthData, clearAuthData } from "../../utils/auth";
import { Home, LogOut, Heart, ArrowLeft, MoreVertical } from "lucide-react";
import PropertyCard from "../shared/PropertyCard";
import "./Dashboard.css";

const Favourites = () => {
  const navigate = useNavigate();
  const { user } = getAuthData();
  const [favourites, setFavourites] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showDropdown, setShowDropdown] = useState(false);

  useEffect(() => {
    fetchFavourites();
  }, []);

  const fetchFavourites = async () => {
    try {
      setLoading(true);
      const response = await getFavourites();
      let props = response.data;

      // Get predictions
      const predictionPayload = props.map((p) => ({
        id: p.id,
        area_sqft: p.area_sqft,
        bhk: p.bhk,
        listing_score: p.listing_score || 5,
        is_furnished: p.is_furnished,
        city_id: p.city_id,
      }));

      const predictions = await predictBulk(predictionPayload);

      props = props.map((p) => {
        const pred = predictions.data.find((pr) => pr.id === p.id);
        return { ...p, predicted_price: pred?.prediction || 0 };
      });

      setFavourites(props);
    } catch (error) {
      console.error("Error fetching favourites:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleRemoveFavourite = async (propertyId) => {
    try {
      await removeFavourite(propertyId);
      setFavourites((prev) => prev.filter((p) => p.id !== propertyId));
    } catch (error) {
      console.error("Error removing favourite:", error);
    }
  };

  const handleLogout = () => {
    clearAuthData();
    navigate("/login");
  };

  const toggleDropdown = () => {
    setShowDropdown(prev => !prev);
  };
  
  // Close the dropdown when an action is clicked
  const handleDropdownAction = (action) => {
    setShowDropdown(false);
    if (action === 'logout') {
      handleLogout();
    }
    // Navigation is handled by the Link components themselves
  };

  return (
    <div className="dashboard">
      <nav className="navbar">
        <div className="navbar-brand">
          <Home size={24} />
          <span>EstateAI - Favourites</span>
        </div>
        <div className="navbar-actions">
          <div className="navbar-user">
            <span>{user?.full_name || user?.email}</span>
          </div>

          {/* Dropdown Toggle Button */}
          <button className="btn-more-dropdown-toggle" onClick={toggleDropdown}>
            More <MoreVertical size={18} />
          </button>

          {/* Dropdown Menu */}
          {showDropdown && (
            <div className="dropdown-menu">
                {/* Back to Dashboard Link */}
                <Link to="/buyer-dashboard" className="dropdown-item" onClick={handleDropdownAction}>
                  <ArrowLeft size={18} /> Back to Dashboard
                </Link>
                
                {/* Logout Button */}
                <button onClick={() => handleDropdownAction('logout')} className="dropdown-item btn-logout-dropdown">
                  <LogOut size={18} /> Logout
                </button>
            </div>
          )}
        </div>
      </nav>

      <div className="dashboard-content">
        <div className="page-header">
          <Heart size={32} className="page-icon" />
          <div>
            <h1 className="dashboard-title">My Favourites</h1>
            <p className="page-subtitle">Properties you've saved for later</p>
          </div>
        </div>

        {loading ? (
          <div className="loading">Loading favourites...</div>
        ) : favourites.length === 0 ? (
          <div className="empty-state">
            <Heart size={64} className="empty-icon" />
            <h3>No Favourites Yet</h3>
            <p>Start exploring properties and add them to your favourites</p>
            <Link to="/buyer-dashboard" className="btn-primary">
              Browse Properties
            </Link>
          </div>
        ) : (
          <div className="properties-grid">
            {favourites.map((property) => (
              <PropertyCard
                key={property.id}
                property={property}
                onFavourite={handleRemoveFavourite}
                isFavourite={true}
                showSellerInfo={true}
                showFavouriteButton={true}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Favourites;
