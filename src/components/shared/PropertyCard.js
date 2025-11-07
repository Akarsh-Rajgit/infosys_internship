// src/components/shared/PropertyCard.js - FIXED VERSION
import React, { useState } from "react";
import {
  MapPin,
  Home,
  Maximize2,
  Phone,
  Mail,
  MessageCircle,
  Heart,
  Star,
} from "lucide-react";
import "./PropertyCard.css";

const CITY_MAP = {
  1: "Bangalore",
  2: "Chennai",
  3: "Delhi",
  4: "Hyderabad",
  5: "Kolkata",
  6: "Lucknow",
  7: "Mumbai",
};

const PropertyCard = ({
  property,
  onFavourite,
  isFavourite,
  showSellerInfo = true,
  showFavouriteButton = false,
  onReview,
}) => {
  const [showReviewForm, setShowReviewForm] = useState(false);
  const [reviewData, setReviewData] = useState({ rating: 5, comment: "" });

  const defaultImage =
    "https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=400&h=300&fit=crop";

  // FIXED: Properly construct image URL
  const getImageUrl = (imgPath) => {
    if (!imgPath) return defaultImage;
    // If it's already a full URL, use it
    if (imgPath.startsWith("http")) return imgPath;
    // Otherwise, construct the URL from the uploads path
    return `http://127.0.0.1:8000${imgPath}`;
  };

  const imageUrl =
    property.images && property.images.length > 0
      ? getImageUrl(property.images[0])
      : defaultImage;

  const handleReviewSubmit = (e) => {
    e.preventDefault();
    if (onReview) {
      onReview(property.id, reviewData);
      setShowReviewForm(false);
      setReviewData({ rating: 5, comment: "" });
    }
  };

  const getStatusBadge = () => {
    const statusColors = {
      active: "bg-green",
      pending: "bg-yellow",
      sold: "bg-red",
    };

    if (property.status && property.status !== "active") {
      return (
        <span
          className={`status-badge ${
            statusColors[property.status] || "bg-gray"
          }`}
        >
          {property.status.toUpperCase()}
        </span>
      );
    }
    return null;
  };

  return (
    <div className="property-card-new">
      <div className="property-layout">
        {/* FIXED: Square thumbnail on left */}
        <div className="property-thumbnail">
          <img
            src={imageUrl}
            alt={property.property_name || "Property"}
            onError={(e) => {
              e.target.src = defaultImage;
            }}
          />
          {property.property_type && (
            <span className="property-type-badge">
              {property.property_type}
            </span>
          )}
          {getStatusBadge()}
          {showFavouriteButton && (
            <button
              className={`favourite-button ${isFavourite ? "active" : ""}`}
              onClick={() => onFavourite(property.id)}
            >
              <Heart size={20} fill={isFavourite ? "currentColor" : "none"} />
            </button>
          )}
        </div>

        {/* Content on right */}
        <div className="property-content-new">
          <h3 className="property-title">
            {property.property_name || "Property"}
          </h3>

          <div className="property-location">
            <MapPin size={16} />
            <span>
              {property.location || CITY_MAP[property.city_id] || "Unknown"}
            </span>
          </div>

          <div className="property-details">
            <div className="property-detail-item">
              <Home size={18} />
              <span>{property.bhk} BHK</span>
            </div>
            <div className="property-detail-item">
              <Maximize2 size={18} />
              <span>{property.area_sqft?.toFixed(0)} sqft</span>
            </div>
          </div>

          {property.predicted_price && (
            <div className="property-price">
              ₹{((property.predicted_price*2 )/ 1000000).toFixed(2)} Crores
            </div>
          )}

          {property.amenities && property.amenities.length > 0 && (
            <div className="property-amenities">
              {property.amenities.slice(0, 3).map((amenity, idx) => (
                <span key={idx} className="amenity-tag">
                  {amenity}
                </span>
              ))}
              {property.amenities.length > 3 && (
                <span className="amenity-tag">
                  +{property.amenities.length - 3}
                </span>
              )}
            </div>
          )}

          {/* FIXED: Show contact details inline */}
          {showSellerInfo &&
            (property.seller_phone || property.seller_email) && (
              <div className="seller-contact-inline">
                <div className="seller-contact-title">Contact Seller</div>
                <div className="contact-info">
                  {property.seller_phone && (
                    <div className="contact-item">
                      <Phone size={14} />
                      <span>{property.seller_phone}</span>
                    </div>
                  )}
                  {property.seller_email && (
                    <div className="contact-item">
                      <Mail size={14} />
                      <span>{property.seller_email}</span>
                    </div>
                  )}
                </div>
                <div className="contact-buttons">
                  {property.seller_phone && (
                    <a
                      href={`tel:${property.seller_phone}`}
                      className="contact-btn"
                    >
                      <Phone size={16} />
                      <span>Call</span>
                    </a>
                  )}
                  {property.seller_email && (
                    <a
                      href={`mailto:${property.seller_email}`}
                      className="contact-btn"
                    >
                      <Mail size={16} />
                      <span>Email</span>
                    </a>
                  )}
                  {property.seller_whatsapp && (
                    <a
                      href={`https://wa.me/${property.seller_whatsapp.replace(
                        /\D/g,
                        ""
                      )}`}
                      className="contact-btn whatsapp"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <MessageCircle size={16} />
                      <span>WhatsApp</span>
                    </a>
                  )}
                </div>
              </div>
            )}

          {/* Review button for buyers */}
          {onReview && !showReviewForm && (
            <button
              className="btn-review"
              onClick={() => setShowReviewForm(true)}
            >
              <Star size={16} /> Write Review
            </button>
          )}

          {/* Review form */}
          {showReviewForm && (
            <div className="review-form">
              <form onSubmit={handleReviewSubmit}>
                <div className="rating-input">
                  <label>Rating:</label>
                  <select
                    value={reviewData.rating}
                    onChange={(e) =>
                      setReviewData({
                        ...reviewData,
                        rating: parseInt(e.target.value),
                      })
                    }
                  >
                    <option value="5">5 Stars</option>
                    <option value="4">4 Stars</option>
                    <option value="3">3 Stars</option>
                    <option value="2">2 Stars</option>
                    <option value="1">1 Star</option>
                  </select>
                </div>
                <textarea
                  placeholder="Share your experience..."
                  value={reviewData.comment}
                  onChange={(e) =>
                    setReviewData({ ...reviewData, comment: e.target.value })
                  }
                  required
                />
                <div className="review-actions">
                  <button type="submit" className="btn-submit">
                    Submit
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowReviewForm(false)}
                    className="btn-cancel"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PropertyCard;
