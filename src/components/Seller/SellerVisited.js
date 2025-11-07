// src/components/Seller/SellerVisited.js - Layout Refined with Separate Boxes
import React, { useState, useEffect, useCallback } from "react";
import { useNavigate, Link } from "react-router-dom";
import { getSellerAnalytics } from "../../services/api";
import { getAuthData, clearAuthData } from "../../utils/auth";
import {
    Home,
    LogOut,
    ArrowLeft,
    Users,
    Heart,
    TrendingUp,
    Phone,
    Mail,
    Star, 
    MessageSquare, 
} from "lucide-react";
import "../Buyer/Dashboard.css"; 

const SellerVisited = () => {
    const navigate = useNavigate();
    const { user } = getAuthData();
    const [analytics, setAnalytics] = useState(null);
    const [loading, setLoading] = useState(true);

    // Helper to calculate total favourites
    const totalFavourites = analytics
        ? Object.values(analytics.property_favourites).reduce((a, b) => a + b, 0)
        : 0;

    // Helper to calculate total reviews
    const totalReviews = analytics ? analytics.reviews.length : 0;

    // Helper to calculate average rating
    const averageRating = totalReviews > 0
        ? (analytics.reviews.reduce((sum, review) => sum + review.rating, 0) / totalReviews).toFixed(1)
        : 0;
        
    // FIXED: Wrap fetchAnalytics in useCallback to satisfy ESLint
    const fetchAnalytics = useCallback(async () => {
        try {
            setLoading(true);
            const response = await getSellerAnalytics();
            setAnalytics(response.data);
        } catch (error) {
            console.error("Error fetching analytics:", error);
        } finally {
            setLoading(false);
        }
    }, []); 

    useEffect(() => {
        fetchAnalytics();
    }, [fetchAnalytics]); 

    const handleLogout = () => {
        clearAuthData();
        navigate("/login");
    };

    return (
        <div className="dashboard">
            <nav className="navbar">
                <div className="navbar-brand">
                    <Home size={24} />
                    <span>EstateAI - Seller Analytics</span>
                </div>
                <div className="navbar-actions">
                    <Link to="/seller-dashboard" className="nav-icon-btn">
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
                    <Users size={32} className="page-icon" />
                    <div>
                        <h1 className="dashboard-title">Visitor Analytics</h1>
                        <p className="page-subtitle">
                            Track buyers interested in your properties, favourites, and reviews
                        </p>
                    </div>
                </div>

                {loading ? (
                    <div className="loading">Loading analytics...</div>
                ) : !analytics ? (
                    <div className="no-data">Failed to load analytics</div>
                ) : (
                    <>
                        {/* Statistics Cards */}
                        <div className="stats-grid">
                            <div className="stat-card">
                                <div className="stat-icon"><Home /></div>
                                <div className="stat-info">
                                    <div className="stat-value">{analytics.total_properties}</div>
                                    <div className="stat-label">Total Properties</div>
                                </div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-icon"><Users /></div>
                                <div className="stat-info">
                                    <div className="stat-value">{analytics.total_unique_buyers}</div>
                                    <div className="stat-label">Interested Buyers</div>
                                </div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-icon"><Heart /></div>
                                <div className="stat-info">
                                    <div className="stat-value">{totalFavourites}</div>
                                    <div className="stat-label">Total Favourites</div>
                                </div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-icon"><TrendingUp /></div>
                                <div className="stat-info">
                                    <div className="stat-value">
                                        {analytics.total_properties > 0 ? (totalFavourites / analytics.total_properties).toFixed(1) : 0}
                                    </div>
                                    <div className="stat-label">Avg Favourites/Property</div>
                                </div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-icon"><MessageSquare /></div>
                                <div className="stat-info">
                                    <div className="stat-value">{totalReviews}</div>
                                    <div className="stat-label">Total Reviews</div>
                                </div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-icon"><Star /></div>
                                <div className="stat-info">
                                    <div className="stat-value">{averageRating}</div>
                                    <div className="stat-label">Average Rating</div>
                                </div>
                            </div>
                        </div>

                        {/* ======================================================= */}
                        {/* NEW SEPARATE BOX 1: Interested Buyers Section */}
                        {/* ======================================================= */}
                        <div className="section-container analytics-box">
                            <h2 className="section-title">
                                <Users size={24} /> Interested Buyers
                            </h2>

                            {analytics.buyers.length === 0 ? (
                                <div className="empty-state">
                                    <Users size={64} className="empty-icon" />
                                    <h3>No Interested Buyers Yet</h3>
                                    <p>When buyers favourite your properties, they'll appear here</p>
                                </div>
                            ) : (
                                <div className="buyers-grid">
                                    {analytics.buyers.map((buyer) => (
                                        <div key={buyer.id} className="buyer-card buyer-details-box">
                                            <div className="buyer-avatar">
                                                {buyer.name.charAt(0).toUpperCase()}
                                            </div>
                                            <div className="buyer-info">
                                                <h3>{buyer.name}</h3>
                                                <div className="buyer-contacts">
                                                    {buyer.email && (
                                                        <a
                                                            href={`mailto:${buyer.email}`}
                                                            className="contact-link"
                                                        >
                                                            <Mail size={16} />
                                                            <span>{buyer.email}</span>
                                                        </a>
                                                    )}
                                                    {buyer.phone && (
                                                        <a
                                                            href={`tel:${buyer.phone}`}
                                                            className="contact-link"
                                                        >
                                                            <Phone size={16} />
                                                            <span>{buyer.phone}</span>
                                                        </a>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                        
                        {/* ======================================================= */}
                        {/* NEW SEPARATE BOX 2: Property Reviews Section */}
                        {/* ======================================================= */}
                        <div className="section-container analytics-box">
                            <h2 className="section-title">
                                <Star size={24} /> Latest Buyer Reviews ({totalReviews})
                            </h2>

                            {analytics.reviews.length === 0 ? (
                                <div className="empty-state small">
                                    <MessageSquare size={48} className="empty-icon" />
                                    <h3>No Reviews Yet</h3>
                                    <p>Buyers can review your properties after viewing them.</p>
                                </div>
                            ) : (
                                <div className="reviews-list">
                                    {analytics.reviews.map((review) => (
                                        <div key={review.id} className="review-card">
                                            <div className="review-header">
                                                <span className="reviewer-name">{review.reviewer_name}</span>
                                                <div className="review-rating">
                                                    {Array(review.rating).fill(0).map((_, i) => <Star key={i} size={16} fill="#FFC107" stroke="#FFC107" />)}
                                                    ({review.rating}/5)
                                                </div>
                                            </div>
                                            <p className="review-comment">"{review.comment}"</p>
                                            <small className="review-date">
                                                Reviewed on: {new Date(review.created_at).toLocaleDateString()}
                                            </small>
                                            <span className="review-property-id">
                                                (Property ID: {review.property_id})
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default SellerVisited;