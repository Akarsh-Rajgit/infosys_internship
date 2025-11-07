// src/pages/Home.js
import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
  Home as HomeIcon,
  MapPin,
  TrendingUp,
  Shield,
  MessageCircle,
  X,
} from "lucide-react";
import "./Home.css";
import IndiaMap from "../components/shared/IndiaMap";
import ChatBot from "../components/shared/ChatBot";

// --- Dummy Data and Placeholder Components ---

const newsData = [
  {
    id: 1,
    title: "Bangalore Housing Prices Surge 15% in Q3",
    snippet:
      "Robust demand and limited supply fuel the real estate boom in the IT capital.",
    date: "Oct 28, 2025",
    link: "https://economictimes.indiatimes.com/wealth/real-estate/news-analysis-articles/bangalore-housing-prices-surge-15-in-q3",
  },
  {
    id: 2,
    title: "RBI Holds Repo Rate: Relief for Home Loan Borrowers",
    snippet:
      "Stability in interest rates expected to maintain positive sentiment in the housing market.",
    date: "Oct 25, 2025",
    link: "https://www.livemint.com/economy/rbi-holds-repo-rate-implications-for-home-loan-borrowers-11654321098.html",
  },
  {
    id: 3,
    title: "Government to Announce New Tax Incentives for First-Time Buyers",
    snippet:
      "The proposed scheme aims to boost affordable housing sales across major cities.",
    date: "Oct 20, 2025",
    link: "https://www.moneycontrol.com/news/business/real-estate/govt-mulls-new-tax-benefits-for-first-time-homebuyers-in-budget-1154321098.html",
  },
];

const NewsSection = ({ news }) => (
  <div className="news-section">
    <div className="section-container">
      <h2 className="section-title">Latest Real Estate News 📰</h2>
      <div className="news-grid">
        {news.map((item) => (
          <div key={item.id} className="news-card">
            <h3 className="news-title">{item.title}</h3>
            <p className="news-snippet">{item.snippet}</p>
            <div className="news-meta">
              <span>{item.date}</span>
              <a
                href={item.link}
                target="_blank"
                rel="noopener noreferrer"
                className="read-more-link"
              >
                Read More
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  </div>
);

const analyticsData = [
  {
    city: "Bangalore",
    listings: "18,500+",
    avgPrice: "₹1.2 Cr",
    growth: "+15%",
  },
  { city: "Mumbai", listings: "22,100+", avgPrice: "₹2.5 Cr", growth: "+8%" },
  { city: "Delhi-NCR", listings: "31,000+", avgPrice: "₹95 L", growth: "+12%" },
  { city: "Chennai", listings: "9,800+", avgPrice: "₹85 L", growth: "+10%" },
];
const AnalyticsBox = ({ data }) => (
  <div className="analytics-box">
    <div className="section-container">
      <h2 className="section-title">Market Insights & Statistics </h2>
      <div className="analytics-grid">
        {data.map((item) => (
          <div key={item.city} className="analytics-card">
            <h4>{item.city}</h4>
            <p>
              Listings: <strong>{item.listings}</strong>
            </p>
            <p>
              Avg. Price: <strong>{item.avgPrice}</strong>
            </p>
            <p className="growth">
              Y-o-Y Growth: <strong>{item.growth}</strong>
            </p>
          </div>
        ))}
      </div>

      {/* Visual Charts - ACTUAL CSS VISUALIZATIONS */}
      <div className="visual-charts-grid">
        {/* Market Share Bar Chart */}
        <div className="chart-container">
          <h3>Market Share Distribution</h3>
          <div className="market-share-chart">
            {/* The height is set inline to represent the data percentage */}
            <div className="chart-bar" style={{ height: "85%" }}>
              <span className="label">Mumbai (28%)</span>
            </div>
            <div
              className="chart-bar"
              style={{ height: "75%", backgroundColor: "#764ba2" }}
            >
              <span className="label">Delhi-NCR (25%)</span>
            </div>
            <div
              className="chart-bar"
              style={{ height: "60%", backgroundColor: "#667eea" }}
            >
              <span className="label">Bangalore (20%)</span>
            </div>
            <div
              className="chart-bar"
              style={{ height: "45%", backgroundColor: "#8a9be0" }}
            >
              <span className="label">Others (15%)</span>
            </div>
          </div>
        </div>

        {/* Inventory vs Demand Donut Chart */}
        <div className="chart-container">
          <h3>Inventory vs. Demand</h3>
          <div className="inventory-demand-chart">
            {/* The conic-gradient property in CSS will use the 70% value */}
            <div className="donut-chart">
              <div className="donut-label">
                70%
                <br />
                Demand
              </div>
            </div>
          </div>
          <div className="chart-legend">
            <p>
              <span style={{ backgroundColor: "#667eea" }}></span> High Demand
              (70%)
            </p>
            <p>
              <span style={{ backgroundColor: "#e0e0e0" }}></span> Available
              Inventory (30%)
            </p>
          </div>
        </div>

        {/* Placeholder for Historical Price Trend (using CSS SVG background) */}
        <div className="chart-container">
          <h3>Historical Price Trend</h3>
          <div className="simple-line-chart">
            {/* CSS will draw a rising line */}
            <div className="line-chart-dots">
              <span className="dot" style={{ top: "80%" }}></span>
              <span className="dot" style={{ top: "30%" }}></span>
              <span className="dot" style={{ top: "50%" }}></span>
              <span className="dot" style={{ top: "10%" }}></span>
            </div>
            <p className="trend-label">
              Prices are trending up 📈 (Last 4 Quarters)
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>
);

// Feature Card Content
const featureContent = {
  "ai-predictions": {
    id: "ai-predictions",
    icon: "🤖",
    title: "AI-Powered Predictions",
    gist: "Get accurate price estimates using advanced machine learning models trained on real market data.",
    fullContent: (
      <>
        <p>
          Our proprietary **Machine Learning models** analyze millions of data
          points, including historical transactions, neighborhood amenities, and
          local economic indicators, to provide you with the most **accurate
          property price predictions** available.
        </p>
        <p>
          This goes beyond simple comparative market analysis. We use **deep
          learning techniques** to detect subtle market shifts and future value,
          giving you a competitive edge whether you are buying or selling.
        </p>
        <p>
          For buyers, this means you can quickly verify if a property is priced
          fairly. For sellers, you get an optimal listing price recommendation
          to maximize your return.
        </p>
        <p>
          The prediction model is **continuously updated** to ensure it reflects
          the latest market dynamics in all major Indian cities we cover.
        </p>
      </>
    ),
  },
  "wide-coverage": {
    id: "wide-coverage",
    icon: "🏘️",
    title: "Wide Coverage",
    gist: "Properties across Bangalore, Mumbai, Delhi, Chennai, Hyderabad, Kolkata, and Lucknow.",
    fullContent: (
      <>
        <p>
          EstateAI provides extensive coverage across **India's 7 major
          metropolitan cities**. Our platform aggregates listings from
          Bangalore, Mumbai, Delhi, Chennai, Hyderabad, Kolkata, and Lucknow.
        </p>
        <p>
          We ensure a **diverse range of properties**, from luxury apartments
          and independent houses to commercial spaces and plots of land, are
          available for you to browse in these high-growth areas.
        </p>
        <p>
          Our expansion plan includes adding more tier-2 and tier-3 cities soon,
          making us a truly national platform for real estate intelligence.
        </p>
        <p>
          This wide net ensures you never miss a potential dream home,
          regardless of the city.
        </p>
      </>
    ),
  },
  "direct-contact": {
    id: "direct-contact",
    icon: "💬",
    title: "Direct Seller Contact",
    gist: "Connect directly with property owners through phone, email, or WhatsApp.",
    fullContent: (
      <>
        <p>
          We believe in **removing the middlemen**. Our platform facilitates
          direct and transparent communication between buyers and verified
          property owners.
        </p>
        <p>
          You can instantly connect via **phone call, direct email, or a
          dedicated WhatsApp chat** for any property you're interested in,
          saving you time and commission fees.
        </p>
        <p>
          This direct channel ensures you get authentic information and can
          negotiate terms swiftly and privately.
        </p>
        <p>
          All contact information is protected until you choose to share it with
          a verified seller.
        </p>
      </>
    ),
  },
  "smart-favourites": {
    id: "smart-favourites",
    icon: "⭐",
    title: "Smart Favourites",
    gist: "Save properties you love and let sellers know you're interested.",
    fullContent: (
      <>
        <p>
          Our Smart Favourites feature is more than just a bookmark. It uses AI
          to track market movement and **notify you of price drops** or similar
          new listings for properties you have favorited.
        </p>
        <p>
          By marking a property as a favourite, you also send a **'signal of
          interest'** to the seller, which can open up direct negotiation
          opportunities.
        </p>
        <p>
          The system intelligently organizes your saved homes, making it easy to
          compare and revisit your top choices.
        </p>
        <p>
          Receive weekly digests summarizing the status of all your favorite
          listings.
        </p>
      </>
    ),
  },
  "seller-analytics": {
    id: "seller-analytics",
    icon: "📊",
    title: "Seller Analytics",
    gist: "Sellers get insights on property performance and interested buyers.",
    fullContent: (
      <>
        <p>
          For our sellers, we provide a robust **analytics dashboard**. Track
          how many views your listing receives, the demographic profile of
          interested buyers, and the number of 'Smart Favourite' signals.
        </p>
        <p>
          These data-driven insights allow you to **optimize your listing price
          and description** for maximum reach and conversion.
        </p>
        <p>
          You also receive **AI-powered suggestions** on the best time to
          connect with potential buyers who show high intent.
        </p>
        <p>
          Understand the full performance lifecycle of your property listing in
          one simple interface.
        </p>
      </>
    ),
  },
  "secure-platform": {
    id: "secure-platform",
    icon: "🔒",
    title: "Secure Platform",
    gist: "Your data is protected with industry-standard security measures.",
    fullContent: (
      <>
        <p>
          Security is paramount. Our platform employs **end-to-end encryption**
          and adheres to the latest data privacy regulations to ensure your
          personal and financial information remains safe.
        </p>
        <p>
          All listings and user profiles undergo a **rigorous verification
          process** to protect against fraud and ensure a trustworthy
          environment for transactions.
        </p>
        <p>
          We use **two-factor authentication (2FA)** for all seller accounts to
          prevent unauthorized access.
        </p>
        <p>
          Our security team continuously monitors for vulnerabilities to keep
          the platform safe 24/7.
        </p>
      </>
    ),
  },
};

const FeatureModal = ({ feature, onClose }) => {
  if (!feature) return null;

  return (
    <div className="feature-modal-overlay">
      <div className="feature-modal-content">
        <button className="modal-close-btn" onClick={onClose}>
          <X size={28} />
        </button>
        <div className="modal-header">
          <div className="feature-icon">{feature.icon}</div>
          <h2>{feature.title}</h2>
        </div>
        <div className="modal-body">{feature.fullContent}</div>
      </div>
    </div>
  );
};

const FeatureCard = ({ content, onOpen }) => (
  <div className="feature-card" onClick={() => onOpen(content)}>
    <div className="feature-icon">{content.icon}</div>
    <h3>{content.title}</h3>
    <p className="feature-gist">{content.gist}...</p>
    <button className="read-more-btn">Read More</button>
  </div>
);

// --- Home Component ---

const Home = () => {
  const [showChat, setShowChat] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState(null); // State for modal

  return (
    <div className="home-page">
      {/* Navigation */}
      <nav className="home-nav">
        <div className="nav-container">
          <div className="nav-brand">
            <HomeIcon size={28} />
            <span>EstateAI</span>
          </div>
          <div className="nav-links">
            <Link to="/login" className="nav-link">
              Login
            </Link>
            <Link to="/register" className="nav-btn">
              Get Started
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section - SIMPLIFIED SPLIT STRUCTURE */}
      <section className="hero-section">
        {/* Left Content Container */}
        <div className="hero-content-left">
          <h1 className="hero-title">
            Find Your <span className="gradient-text">Dream Home</span> with
            <br />
            AI-Powered Insights
          </h1>
          <p className="hero-description">
            Discover the perfect property with intelligent price predictions,
            comprehensive listings across India's major cities, and direct
            seller connections.
          </p>
          <div className="hero-buttons">
            <Link to="/register" className="btn-primary-large">
              Start Exploring
            </Link>
            <Link to="/login" className="btn-secondary-large">
              Sign In
            </Link>
          </div>

          <div className="features-grid-mini">
            <div className="feature-mini">
              <TrendingUp size={24} />
              <span>AI Price Predictions</span>
            </div>
            <div className="feature-mini">
              <MapPin size={24} />
              <span>7 Major Cities</span>
            </div>
            <div className="feature-mini">
              <Shield size={24} />
              <span>Verified Listings</span>
            </div>
          </div>
        </div>

        {/* Right Image Container (using <img> tag) */}
        <div className="hero-image-right">
          <img src="https://i.pinimg.com/1200x/6a/6e/6d/6a6e6d129cc326396bbee54fb29c23d2.jpg" alt="Modern luxury home" className="hero-main-image"/>
        </div>
      </section>

      {/* Map Section */}
      <section className="map-section">
        <div className="section-container">
          <h2 className="section-title">Explore Properties Across India</h2>
          <p className="section-subtitle">
            Click on a city to discover available properties in that location
          </p>
          <IndiaMap />
        </div>
      </section>

      {/* Analytics Box (with Charts) */}
      <AnalyticsBox data={analyticsData} />

      {/* News Section */}
      <NewsSection news={newsData} />

      {/* Features Section */}
      <section className="features-section">
        <div className="section-container">
          <h2 className="section-title">Why Choose EstateAI?</h2>
          <div className="features-grid">
            {Object.values(featureContent).map((content) => (
              <FeatureCard
                key={content.id}
                content={content}
                onOpen={setSelectedFeature}
              />
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="cta-content">
          <h2>Ready to Find Your Perfect Property?</h2>
          <p>Join thousands of buyers and sellers using EstateAI</p>
          <Link to="/register" className="cta-button">
            Get Started Free
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="home-footer">
        <div className="footer-content">
          <div className="footer-brand">
            <HomeIcon size={24} />
            <span>EstateAI</span>
          </div>
          <p>
            © 2024 EstateAI. Powered by AI for smarter real estate decisions.
          </p>
        </div>
      </footer>

      {/* Floating Chat Button/Widget */}
      <button className="chat-fab" onClick={() => setShowChat(!showChat)}>
        {showChat ? <X size={24} /> : <MessageCircle size={24} />}
      </button>
      {showChat && <ChatBot onClose={() => setShowChat(false)} />}

      {/* Feature Modal (Full-Screen) */}
      <FeatureModal
        feature={selectedFeature}
        onClose={() => setSelectedFeature(null)}
      />
    </div>
  );
};

export default Home;
