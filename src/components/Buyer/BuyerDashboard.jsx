// src/components/Buyer/BuyerDashboard.js - FIXED WITH REVIEWS AND CHARTS
import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { getProperties, predictBulk, addFavourite, removeFavourite, getFavourites, submitReview } from '../../services/api';
import { getAuthData, clearAuthData } from '../../utils/auth';
import { Home, LogOut, Heart, Calculator, MessageCircle, X, Search, BarChart as BarChartIcon, MoreVertical} from 'lucide-react';
import PropertyCard from '../shared/PropertyCard';
import ChatBot from '../shared/ChatBot';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell,
} from 'recharts'; // IMPORT RECHARTS COMPONENTS
import './Dashboard.css';

const CITY_MAP = {
  1: 'Bangalore', 2: 'Chennai', 3: 'Delhi', 4: 'Hyderabad',
  5: 'Kolkata', 6: 'Lucknow', 7: 'Mumbai'
};

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#A288FF', '#FF66AA', '#00E0C0'];

const BuyerDashboard = () => {
  const navigate = useNavigate();
  const { user } = getAuthData();
  const [properties, setProperties] = useState([]);
  const [filteredProperties, setFilteredProperties] = useState([]);
  const [favourites, setFavourites] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showChat, setShowChat] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [filters, setFilters] = useState({
    city_id: '',
    bhk: '',
    area_sqft: '',
  });

  // Chart State
  const [priceComparison, setPriceComparison] = useState([]);
  const [cityDistribution, setCityDistribution] = useState([]);
  const [bhkDistribution, setBHKDistribution] = useState([]);

  // FIXED: Data processing logic wrapped in useCallback
  const processChartData = useCallback((props) => {
    // 1. Price Comparison (Top 10 most expensive predicted prices)
    const sortedByPrice = [...props].sort((a, b) => b.predicted_price - a.predicted_price);
    const top10 = sortedByPrice.slice(0, 10).map(p => ({
      name: `${p.bhk} BHK, ${CITY_MAP[p.city_id] || 'N/A'}`,
      price: parseFloat((((p.predicted_price * 2)/ 1000000)).toFixed(2)), // Convert to Lakhs and format
    }));
    setPriceComparison(top10);

    // 2. City Distribution
    const cityCounts = props.reduce((acc, p) => {
      const cityName = CITY_MAP[p.city_id] || 'Other';
      acc[cityName] = (acc[cityName] || 0) + 1;
      return acc;
    }, {});
    const cityData = Object.entries(cityCounts).map(([name, value]) => ({ name, value }));
    setCityDistribution(cityData);

    // 3. BHK Distribution
    const bhkCounts = props.reduce((acc, p) => {
      const bhkLabel = `${p.bhk} BHK`;
      acc[bhkLabel] = (acc[bhkLabel] || 0) + 1;
      return acc;
    }, {});
    const bhkData = Object.entries(bhkCounts).map(([name, value]) => ({ name, value })).sort((a, b) => {
        const numA = parseInt(a.name.split(' ')[0]);
        const numB = parseInt(b.name.split(' ')[0]);
        return numA - numB;
    });
    setBHKDistribution(bhkData);
  }, []); // Dependencies are stable setters

  // FIXED: fetchProperties wrapped in useCallback
  const fetchProperties = useCallback(async () => {
    try {
      setLoading(true);
      const response = await getProperties();
      let props = response.data;

      const predictionPayload = props.map(p => ({
        id: p.id,
        area_sqft: p.area_sqft,
        bhk: p.bhk,
        listing_score: p.listing_score || 5,
        is_furnished: p.is_furnished,
        city_id: p.city_id
      }));

      // Check if there are properties to predict
      if (predictionPayload.length > 0) {
          const predictions = await predictBulk(predictionPayload);
          
          props = props.map(p => {
            const pred = predictions.data.find(pr => pr.id === p.id);
            // Predicted price is returned in currency units
            return { ...p, predicted_price: pred?.prediction || 0 }; 
          });
      }
      

      setProperties(props);
      setFilteredProperties(props);
      processChartData(props); // Process data for charts
    } catch (error) {
      console.error('Error fetching properties:', error);
    } finally {
      setLoading(false);
    }
  }, [processChartData]); // Dependency array includes the stable processChartData

  // FIXED: fetchFavourites wrapped in useCallback
  const fetchFavourites = useCallback(async () => {
    try {
      const response = await getFavourites();
      setFavourites(response.data.map(p => p.id));
    } catch (error) {
      console.error('Error fetching favourites:', error);
    }
  }, []); // Dependencies are stable api calls and setters

  // FIXED: useEffect hook now includes the stable functions
  useEffect(() => {
    fetchProperties();
    fetchFavourites();
  }, [fetchProperties, fetchFavourites]); // Dependencies are now the stable functions

  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilters(prev => ({ ...prev, [name]: value }));
  };

  const applyFilters = () => {
    let filtered = [...properties];

    if (filters.city_id) {
      filtered = filtered.filter(p => p.city_id === parseInt(filters.city_id));
    }
    if (filters.bhk) {
      filtered = filtered.filter(p => p.bhk === parseInt(filters.bhk));
    }
    if (filters.area_sqft) {
      filtered = filtered.filter(p => p.area_sqft <= parseFloat(filters.area_sqft));
    }

    setFilteredProperties(filtered);
  };

  const clearFilters = () => {
    setFilters({ city_id: '', bhk: '', area_sqft: '' });
    setFilteredProperties(properties);
  };

  const handleFavourite = async (propertyId) => {
    try {
      if (favourites.includes(propertyId)) {
        await removeFavourite(propertyId);
        setFavourites(prev => prev.filter(id => id !== propertyId));
      } else {
        await addFavourite(propertyId);
        setFavourites(prev => [...prev, propertyId]);
      }
    } catch (error) {
      console.error('Error updating favourite:', error);
    }
  };

  const handleReview = async (propertyId, reviewData) => {
    try {
      await submitReview({
        property_id: propertyId,
        rating: reviewData.rating,
        comment: reviewData.comment
      });
      alert('Review submitted successfully!');
    } catch (error) {
      console.error('Error submitting review:', error);
      alert(error.response?.data?.detail || 'Failed to submit review');
    }
  };

  const handleLogout = () => {
    clearAuthData();
    navigate('/login');
  };
  
  // Toggle the dropdown
  function toggleDropdown() {
    setShowDropdown(prev => !prev);
  }
  
  // Close the dropdown when an action is clicked
  const handleDropdownAction = (action) => {
    setShowDropdown(false);
    if (action === 'logout') {
      handleLogout();
    }
    // Other actions (like navigation) will be handled by the Link components
  };

  return (
    <div className="dashboard">
      <nav className="navbar">
        <div className="navbar-brand">
          <Home size={24} />
          <span>EstateAI - Buyer</span>
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
                {/* Favorites Link */}
                <Link to="/favourites" className="dropdown-item" onClick={handleDropdownAction}>
                  <Heart size={18} /> Favorites 
                  {favourites.length > 0 && (
                    <span className="badge-dropdown">{favourites.length}</span>
                  )}
                </Link>
                
                {/* EMI Calculator Link */}
                <Link to="/emi-calculator" className="dropdown-item" onClick={handleDropdownAction}>
                  <Calculator size={18} /> EMI Calculator
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
        <h1 className="dashboard-title">Discover Properties</h1>

        {/* Filter Section */}
        <div className="filter-section">
          <h2><Search size={20} /> Filter Properties</h2>
          <div className="filter-grid">
            <div className="filter-item">
              <label>City</label>
              <select name="city_id" value={filters.city_id} onChange={handleFilterChange}>
                <option value="">All Cities</option>
                {Object.entries(CITY_MAP).map(([id, name]) => (
                  <option key={id} value={id}>{name}</option>
                ))}
              </select>
            </div>
            <div className="filter-item">
              <label>BHK</label>
              <select name="bhk" value={filters.bhk} onChange={handleFilterChange}>
                <option value="">All</option>
                <option value="1">1 BHK</option>
                <option value="2">2 BHK</option>
                <option value="3">3 BHK</option>
                <option value="4">4+ BHK</option>
              </select>
            </div>
            <div className="filter-item">
              <label>Max Area (sqft)</label>
              <input
                type="number"
                name="area_sqft"
                value={filters.area_sqft}
                onChange={handleFilterChange}
                placeholder="e.g., 2000"
              />
            </div>
            <div className="filter-buttons">
              <button onClick={applyFilters} className="btn-primary">Apply</button>
              <button onClick={clearFilters} className="btn-secondary">Clear</button>
            </div>
          </div>
        </div>
        
        {/* Charts Section */}
        <h2 className="charts-title"><BarChartIcon size={20} /> Market Insights</h2>
        <div className="charts-grid">
          <div className="chart-card">
            <h3>Price Comparison (Top 10)</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={priceComparison}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis label={{ value: 'Price (Crores)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="price" fill="#667eea" name="Price (₹ Lakhs)" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <h3>City Distribution</h3>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={cityDistribution}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {cityDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <h3>BHK Distribution</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={bhkDistribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" fill="#764ba2" name="Count" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Properties Grid */}
        {loading ? (
          <div className="loading">Loading properties...</div>
        ) : filteredProperties.length === 0 ? (
          <div className="no-data">No properties found matching your filters.</div>
        ) : (
          <div className="properties-grid">
            {filteredProperties.map((property) => (
              <PropertyCard
                key={property.id}
                property={property}
                onFavourite={handleFavourite}
                isFavourite={favourites.includes(property.id)}
                showSellerInfo={true}
                showFavouriteButton={true}
                onReview={handleReview}
              />
            ))}
          </div>
        )}
      </div>

      {/* Floating Chat Button */}
      <button className="chat-fab" onClick={() => setShowChat(!showChat)}>
        {showChat ? <X size={24} /> : <MessageCircle size={24} />}
      </button>

      {/* Chat Widget */}
      {showChat && <ChatBot onClose={() => setShowChat(false)} />}
    </div>
  );
};

export default BuyerDashboard;