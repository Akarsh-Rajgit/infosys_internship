// src/components/Seller/SellerDashboard.jsx - FIXED VERSION WITH PREDICTED PRICE
import React, { useState, useEffect, useCallback } from "react";
import { useNavigate, Link } from "react-router-dom";
// Import predictBulk API service
import { getProperties, createProperty, uploadImage, predictBulk } from "../../services/api"; 
import { getAuthData, clearAuthData } from "../../utils/auth";
import {
  Home,
  Plus,
  LogOut,
  Eye,
  MessageCircle,
  X,
  Upload,
  XCircle,
  MoreVertical,
} from "lucide-react";
import PropertyCard from "../shared/PropertyCard";
import ChatBot from "../shared/ChatBot";
import "../Buyer/Dashboard.css";

const CITY_MAP = {
  1: "Bangalore",
  2: "Chennai",
  3: "Delhi",
  4: "Hyderabad",
  5: "Kolkata",
  6: "Lucknow",
  7: "Mumbai",
};

const SellerDashboard = () => {
  const navigate = useNavigate();
  const { user } = getAuthData();
  const [properties, setProperties] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAddForm, setShowAddForm] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [uploadingImages, setUploadingImages] = useState(false);
  const [formData, setFormData] = useState({
    property_name: "",
    city_id: "",
    location: "",
    area_sqft: "",
    bhk: "",
    is_furnished: false,
    listing_score: 5,
    property_type: "apartment",
    amenities: "",
    seller_phone: user?.phone || "",
    seller_email: user?.email || "",
    seller_whatsapp: user?.whatsapp || "",
    images: [],
    status: "active", // NEW
  });
  const [submitLoading, setSubmitLoading] = useState(false);
  const [message, setMessage] = useState({ type: "", text: "" });

  // FIXED: Wrap fetchProperties in useCallback to stabilize the function
  const fetchProperties = useCallback(async () => {
    try {
      setLoading(true);
      // 1. Fetch properties listed by this seller
      const response = await getProperties({ seller_id: user.id }); 
      let props = response.data;
      
      if (props.length > 0) {
          // 2. Prepare prediction payload
          const predictionPayload = props.map(p => ({
              id: p.id,
              area_sqft: p.area_sqft,
              bhk: p.bhk,
              listing_score: p.listing_score || 5,
              is_furnished: p.is_furnished,
              city_id: p.city_id
          }));

          // 3. Get predictions
          const predictionsResponse = await predictBulk(predictionPayload);
          const predictions = predictionsResponse.data;
          
          // 4. Map predicted price back to properties
          props = props.map(p => {
              const pred = predictions.find(pr => pr.id === p.id);
              return { ...p, predicted_price: pred?.prediction || 0 }; 
          });
      }
      
      setProperties(props);
    } catch (error) {
      console.error("Error fetching properties or predictions:", error);
      setMessage({ type: "error", text: "Failed to load properties or predicted prices." });
    } finally {
      setLoading(false);
    }
  }, [user.id]); // Dependency array includes user.id

  // FIXED: useEffect now includes the stable fetchProperties function
  useEffect(() => {
    fetchProperties();
  }, [fetchProperties]); // Dependency array includes fetchProperties

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handleImageUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    setUploadingImages(true);
    const uploadedUrls = [];

    try {
      for (const file of files) {
        const formDataImg = new FormData();
        formDataImg.append("file", file);

        const response = await uploadImage(formDataImg);
        uploadedUrls.push(response.data.url);
      }

      setFormData((prev) => ({
        ...prev,
        images: [...prev.images, ...uploadedUrls],
      }));

      setMessage({
        type: "success",
        text: `${files.length} image(s) uploaded successfully!`,
      });
    } catch (error) {
      console.error("Error uploading images:", error);
      setMessage({ type: "error", text: "Failed to upload images" });
    } finally {
      setUploadingImages(false);
      setTimeout(() => setMessage({ type: "", text: "" }), 3000);
    }
  };

  const removeImage = (index) => {
    setFormData((prev) => ({
      ...prev,
      images: prev.images.filter((_, i) => i !== index),
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitLoading(true);
    setMessage({ type: "", text: "" });

    try {
      const propertyData = {
        property_name: formData.property_name,
        city_id: parseInt(formData.city_id),
        location: formData.location,
        area_sqft: parseFloat(formData.area_sqft),
        bhk: parseInt(formData.bhk),
        is_furnished: formData.is_furnished,
        listing_score: parseFloat(formData.listing_score),
        property_type: formData.property_type,
        amenities: formData.amenities
          .split(",")
          .map((a) => a.trim())
          .filter((a) => a),
        seller_phone: formData.seller_phone,
        seller_email: formData.seller_email,
        seller_whatsapp: formData.seller_whatsapp,
        images: formData.images,
        seller_id: user.id,
        status: formData.status, // NEW
      };

      await createProperty(propertyData);
      setMessage({ type: "success", text: "Property added successfully!" });
      setFormData({
        property_name: "",
        city_id: "",
        location: "",
        area_sqft: "",
        bhk: "",
        is_furnished: false,
        listing_score: 5,
        property_type: "apartment",
        amenities: "",
        seller_phone: user?.phone || "",
        seller_email: user?.email || "",
        seller_whatsapp: user?.whatsapp || "",
        images: [],
        status: "active",
      });
      setShowAddForm(false);
      fetchProperties();
    } catch (error) {
      console.error("Error adding property:", error);
      setMessage({
        type: "error",
        text: error.response?.data?.detail || "Failed to add property",
      });
    } finally {
      setSubmitLoading(false);
    }
  };

  const handleLogout = () => {
    clearAuthData();
    navigate("/login");
  };

  // Toggle the dropdown
  const toggleDropdown = () => {
    setShowDropdown(prev => !prev);
  };
  
  // Close the dropdown when an action is clicked
  const handleDropdownAction = (action) => {
    setShowDropdown(false);
    if (action === 'logout') {
      handleLogout();
    }
  };

  return (
    <div className="dashboard">
      <nav className="navbar">
        <div className="navbar-brand">
          <Home size={24} />
          <span>EstateAI - Seller</span>
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
                {/* Visited and Reviewed Link */}
                <Link to="/seller-visited" className="dropdown-item" onClick={handleDropdownAction}>
                  <Eye size={18} /> Visited & Reviewed
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
        <div className="dashboard-header">
          <h1 className="dashboard-title">My Properties</h1>
          <button
            onClick={() => setShowAddForm(!showAddForm)}
            className="btn-add"
          >
            <Plus size={20} /> Add Property
          </button>
        </div>

        {message.text && (
          <div className={`message ${message.type}`}>{message.text}</div>
        )}

        {/* Add Property Form */}
        {showAddForm && (
          <div className="form-section">
            <h2>Add New Property</h2>
            <form onSubmit={handleSubmit} className="property-form">
              <div className="form-grid">
                <div className="form-group">
                  <label>Property Name *</label>
                  <input
                    type="text"
                    name="property_name"
                    value={formData.property_name}
                    onChange={handleInputChange}
                    required
                    placeholder="e.g., Luxury Villa in Whitefield"
                  />
                </div>

                <div className="form-group">
                  <label>City *</label>
                  <select
                    name="city_id"
                    value={formData.city_id}
                    onChange={handleInputChange}
                    required
                  >
                    <option value="">Select City</option>
                    {Object.entries(CITY_MAP).map(([id, name]) => (
                      <option key={id} value={id}>
                        {name}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="form-group">
                  <label>Location *</label>
                  <input
                    type="text"
                    name="location"
                    value={formData.location}
                    onChange={handleInputChange}
                    required
                    placeholder="e.g., Whitefield, Near ITPL"
                  />
                </div>

                <div className="form-group">
                  <label>Property Type *</label>
                  <select
                    name="property_type"
                    value={formData.property_type}
                    onChange={handleInputChange}
                    required
                  >
                    <option value="apartment">Apartment</option>
                    <option value="villa">Villa</option>
                    <option value="plot">Plot</option>
                    <option value="house">Independent House</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Status *</label>
                  <select
                    name="status"
                    value={formData.status}
                    onChange={handleInputChange}
                    required
                  >
                    <option value="pending">Pending</option>
                    <option value="active">Active</option>
                    <option value="sold">Sold</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Area (sqft) *</label>
                  <input
                    type="number"
                    name="area_sqft"
                    value={formData.area_sqft}
                    onChange={handleInputChange}
                    required
                    min="100"
                    step="0.01"
                    placeholder="e.g., 1200"
                  />
                </div>

                <div className="form-group">
                  <label>BHK *</label>
                  <select
                    name="bhk"
                    value={formData.bhk}
                    onChange={handleInputChange}
                    required
                  >
                    <option value="">Select BHK</option>
                    <option value="1">1 BHK</option>
                    <option value="2">2 BHK</option>
                    <option value="3">3 BHK</option>
                    <option value="4">4 BHK</option>
                    <option value="5">5+ BHK</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Amenities (comma separated)</label>
                  <input
                    type="text"
                    name="amenities"
                    value={formData.amenities}
                    onChange={handleInputChange}
                    placeholder="e.g., Parking, Gym, Pool, Garden"
                  />
                </div>

                <div className="form-group">
                  <label>Listing Score (1-10)</label>
                  <input
                    type="number"
                    name="listing_score"
                    value={formData.listing_score}
                    onChange={handleInputChange}
                    min="1"
                    max="10"
                    step="0.1"
                    placeholder="5.0"
                  />
                </div>

                <div className="form-group">
                  <label>Contact Phone</label>
                  <input
                    type="tel"
                    name="seller_phone"
                    value={formData.seller_phone}
                    onChange={handleInputChange}
                    placeholder="+91 1234567890"
                  />
                </div>

                <div className="form-group">
                  <label>Contact Email</label>
                  <input
                    type="email"
                    name="seller_email"
                    value={formData.seller_email}
                    onChange={handleInputChange}
                    placeholder="your.email@example.com"
                  />
                </div>

                <div className="form-group">
                  <label>WhatsApp Number</label>
                  <input
                    type="tel"
                    name="seller_whatsapp"
                    value={formData.seller_whatsapp}
                    onChange={handleInputChange}
                    placeholder="+91 1234567890"
                  />
                </div>

                <div className="form-group checkbox-group">
                  <label>
                    <input
                      type="checkbox"
                      name="is_furnished"
                      checked={formData.is_furnished}
                      onChange={handleInputChange}
                    />
                    <span>Furnished</span>
                  </label>
                </div>
              </div>

              {/* Image Upload Section */}
              <div className="image-upload-section">
                <label>Property Images</label>
                <div className="upload-area">
                  <input
                    type="file"
                    accept="image/*"
                    multiple
                    onChange={handleImageUpload}
                    id="image-upload"
                    style={{ display: "none" }}
                  />
                  <label htmlFor="image-upload" className="upload-button">
                    <Upload size={24} />
                    <span>
                      {uploadingImages ? "Uploading..." : "Upload Images"}
                    </span>
                  </label>
                </div>

                {formData.images.length > 0 && (
                  <div className="image-preview-grid">
                    {formData.images.map((img, idx) => (
                      <div key={idx} className="image-preview">
                        <img
                          src={`http://127.0.0.1:8000${img}`} 
                          alt={`Preview ${idx + 1}`}
                        />
                        <button
                          type="button"
                          className="remove-image"
                          onClick={() => removeImage(idx)}
                        >
                          <XCircle size={20} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="form-actions">
                <button
                  type="submit"
                  className="btn-primary"
                  disabled={submitLoading || uploadingImages}
                >
                  {submitLoading ? "Adding..." : "Add Property"}
                </button>
                <button
                  type="button"
                  onClick={() => setShowAddForm(false)}
                  className="btn-secondary"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Properties Grid */}
        {loading ? (
          <div className="loading">Loading properties...</div>
        ) : properties.length === 0 ? (
          <div className="empty-state">
            <Home size={64} className="empty-icon" />
            <h3>No Properties Listed Yet</h3>
            <p>Start by adding your first property</p>
            <button
              onClick={() => setShowAddForm(true)}
              className="btn-primary"
            >
              <Plus size={20} /> Add Property
            </button>
          </div>
        ) : (
          <div className="properties-grid">
            {properties.map((property) => (
              <PropertyCard
                key={property.id}
                property={property}
                showSellerInfo={false}
                showFavouriteButton={false}
                // NEW PROP: Pass predictedPrice to PropertyCard
                predictedPrice={property.predicted_price} 
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

export default SellerDashboard;