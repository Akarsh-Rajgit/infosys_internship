// src/services/api.js - FIXED VERSION
import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Add token to requests automatically
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Auth APIs
export const register = (userData) => api.post("/api/register", userData);
export const login = (credentials) => api.post("/api/login", credentials);

// Property APIs
export const getProperties = (filters = {}) => {
  const params = new URLSearchParams();
  if (filters.city_id) params.append("city_id", filters.city_id);
  if (filters.bhk) params.append("bhk", filters.bhk);
  if (filters.area_sqft) params.append("area_sqft", filters.area_sqft);
  if (filters.seller_id) params.append("seller_id", filters.seller_id);

  return api.get(`/api/properties?${params.toString()}`);
};

export const createProperty = (propertyData) =>
  api.post("/api/properties", propertyData);

// Image Upload
export const uploadImage = (formData) => {
  const token = localStorage.getItem("token");
  return axios.post(`${API_BASE_URL}/api/upload-image`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
      Authorization: `Bearer ${token}`,
    },
  });
};

// Predictions
export const predictPrice = (propertyData) =>
  api.post("/api/predict42", propertyData);
export const predictBulk = (properties) =>
  api.post("/api/predict_bulk", properties);

// Favourites - FIXED: Token passed via header, not query param
export const addFavourite = (propertyId) => {
  return api.post("/api/favourites", { property_id: propertyId });
};

export const getFavourites = () => {
  return api.get("/api/favourites");
};

export const removeFavourite = (propertyId) => {
  return api.delete(`/api/favourites/${propertyId}`);
};

// Seller Analytics - FIXED: Token passed via header
export const getSellerAnalytics = () => {
  return api.get("/api/seller/analytics");
};

// Reviews - NEW
export const submitReview = (reviewData) => {
  return api.post("/api/reviews", reviewData);
};

export const getPropertyReviews = (propertyId) => {
  return api.get(`/api/reviews/${propertyId}`);
};

// Chat - FIXED: Better error handling
export const sendChatMessage = (message) => {
  return api.post("/chatbot", { message }).catch((error) => {
    console.error("Chat API failed:", error);
    // Return fallback response
    return { data: { response: getFallbackResponse(message) } };
  });
};

// Fallback responses for when chatbot is down
function getFallbackResponse(message) {
  const msg = message.toLowerCase();

  const faqs = {
    rera: "RERA (Real Estate Regulatory Authority) registration ensures your property is legally compliant. Check if the property has a valid RERA number before purchasing.",
    loan: "Home loans are available from various banks with interest rates ranging from 8-9% p.a. Check our EMI Calculator to estimate your monthly payments.",
    documents:
      "Essential documents include: Title Deed, Sale Agreement, Encumbrance Certificate, Property Tax Receipts, Completion Certificate, and Occupancy Certificate.",
    price:
      "Our AI predicts property prices based on location, size, amenities, and market trends. The predicted price is an estimate - actual prices may vary.",
    seller:
      "As a seller, you can list properties with images, track interested buyers, and view analytics on your dashboard.",
    buyer:
      "As a buyer, you can browse properties, filter by location/BHK/area, save favourites, and contact sellers directly.",
    contact:
      "You can contact sellers via Call, Email, or WhatsApp buttons on each property card.",
    favourite:
      "Click the heart icon on any property to add it to your favourites. Sellers can see when buyers favourite their properties.",
    investment:
      "Consider factors like location, infrastructure development, rental yield, and appreciation potential. Diversify your real estate portfolio.",
    tax: "Property tax varies by city. Capital gains tax applies when selling. Consult a tax advisor for specific guidance.",
  };

  for (const [key, response] of Object.entries(faqs)) {
    if (msg.includes(key)) {
      return response;
    }
  }

  return "I'm here to help with real estate queries! Ask me about RERA registration, home loans, property documents, price predictions, or how to use this platform. You can also check our EMI Calculator for loan planning.";
}

export default api;
