import React, { useState } from "react";
import "./IndiaMap.css";

// Assuming you've fixed the path issue and this correctly points to your map image
const IndiaOutlineMap = '/india-outline-map.png'; 

const IndiaMap = () => {
    const [hoveredCity, setHoveredCity] = useState(null);

    // FINAL FIX: Adjusted coordinates (x, y) for precise geographic placement on the provided map image.
    const cities = [
        // City markers (x = Left/Right, y = Top/Bottom)
        { id: 3, name: "Delhi", x: 37, y: 27, color: "#f093fb" },      // North (Pink) - Moved left slightly
        { id: 6, name: "Lucknow", x: 48.5, y: 36.5, color: "#fa709a" },    // North-Central (Red/Pink)
        { id: 5, name: "Kolkata", x: 62.5, y: 45.5, color: "#43e97b" },    // East (Green)
        { id: 7, name: "Mumbai", x: 28.5, y: 59.5, color: "#fee140" },     // West (Yellow) - Moved left slightly
        { id: 4, name: "Hyderabad", x: 44.5, y: 62.5, color: "#4facfe" },   // Central/South (Blue) - Moved up slightly
        { id: 1, name: "Bangalore", x: 38.5, y: 72.5, color: "#667eea" },  // South (Light Purple/Blue) - Adjusted down/right slightly
        { id: 2, name: "Chennai", x: 45.5, y: 73.5, color: "#764ba2" },    // Southeast (Dark Purple) - Adjusted down/right
    ];

    return (
        <div className="india-map-container">
            <div className="map-image-wrapper">
                <img 
                    src={IndiaOutlineMap} 
                    alt="Outline Map of India" 
                    className="india-map-background-image" 
                />

                {/* City markers positioned over the image */}
                {cities.map((city) => {
                    const isHovered = hoveredCity === city.id;
                    return (
                        <div
                            key={city.id}
                            className="city-marker-container"
                            style={{ 
                                left: `${city.x}%`, 
                                top: `${city.y}%`, 
                                '--marker-color': city.color 
                            }}
                            onMouseEnter={() => setHoveredCity(city.id)}
                            onMouseLeave={() => setHoveredCity(null)}
                            onFocus={() => setHoveredCity(city.id)}
                            onBlur={() => setHoveredCity(null)}
                            tabIndex="0" 
                        >
                            {/* subtle pulse behind the marker */}
                            <div className={`city-pulse ${isHovered ? "pulse-active" : ""}`} />

                            {/* white-ringed colored marker */}
                            <div className="city-marker" />

                            {/* visible label */}
                            {isHovered && (
                                <div className="city-label">
                                    {city.name}
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>

            {/* legend (mirrors the city array) */}
            <div className="city-legend">
                {cities.map((c) => (
                    <button
                        key={c.id}
                        className={`legend-item ${hoveredCity === c.id ? "active" : ""}`}
                        onMouseEnter={() => setHoveredCity(c.id)}
                        onMouseLeave={() => setHoveredCity(null)}
                        onFocus={() => setHoveredCity(c.id)}
                        onBlur={() => setHoveredCity(null)}
                        type="button"
                        aria-label={`Highlight ${c.name}`}
                    >
                        <span
                            className="legend-color"
                            style={{ background: c.color, boxShadow: "0 1px 3px rgba(0,0,0,0.12)" }}
                        />
                        <span className="legend-text">{c.name}</span>
                    </button>
                ))}
            </div>
        </div>
    );
};

export default IndiaMap;