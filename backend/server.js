const express = require('express');
const cors = require('cors');

const app = express();
const PORT = 5001;

// Middleware to parse incoming JSON payloads from Python
app.use(express.json());
app.use(cors());

// The exact endpoint Python is looking for
app.post('/api/shelf-events', (req, res) => {
    // Extract the shopper count payload sent by your vision daemon
    const { count } = req.body;
    
    // Log the AI data to the console so we can prove it works
    console.log(`[${new Date().toLocaleTimeString()}] 📥 AI Alert: ${count} shopper(s) detected at the shelf!`);
    
    // Send a 200 OK back to Python so it knows the message was safely received
    res.status(200).json({ success: true, message: "Telemetry received" });
});

// Boot up the server
app.listen(PORT, () => {
    console.log(`⚡ MERN Backend active and listening on http://localhost:${PORT}`);
});