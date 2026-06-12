const express = require('express');
const cors = require('cors');

const app = express();
app.use(express.json());
app.use(cors());

const PORT = 5001;

// --- IN-MEMORY DATABASE (For Testing) ---
// We will swap this out for MongoDB Atlas later
let globalDatabase = [];

// --- THE AGGREGATOR CONFIG ---
// If two events happen at the exact same shelf within 3 seconds of each other,
// the server assumes the AI fragmented the ID and merges them into one session.
const MERGE_WINDOW_MS = 3000; 

app.post('/api/shelf-events', (req, res) => {
    const { shopper_id, shelf, dwell_time } = req.body;
    
    // Safety check: ignore empty payloads
    if (!shelf || !dwell_time) {
        return res.status(400).json({ error: "Invalid payload" });
    }

    const currentTime = Date.now();
    let wasMerged = false;

    // --- ID FRAGMENTATION FIX (The Merger) ---
    // Look backwards through our database to see if someone was JUST at this shelf
    for (let i = globalDatabase.length - 1; i >= 0; i--) {
        const recentEvent = globalDatabase[i];
        
        const timeSinceLastEvent = currentTime - recentEvent.last_seen_timestamp;

        if (recentEvent.shelf === shelf && timeSinceLastEvent <= MERGE_WINDOW_MS) {
            console.log(`\n🔗 FRAGMENTATION DETECTED! Merging new ID ${shopper_id} into previous session.`);
            
            // Add the new fragmented time to the master time
            recentEvent.total_dwell_time = parseFloat((recentEvent.total_dwell_time + dwell_time).toFixed(1));
            
            // Reset the clock so the window stays open
            recentEvent.last_seen_timestamp = currentTime; 
            
            console.log(`📊 UPDATED ${shelf} SESSION -> Total Time: ${recentEvent.total_dwell_time}s`);
            wasMerged = true;
            break;
        }
    }

    // --- GENUINE NEW SHOPPER LOGIC ---
    if (!wasMerged) {
        console.log(`\n📥 NEW SHOPPER EVENT: ID ${shopper_id} at ${shelf} for ${dwell_time}s`);
        
        globalDatabase.push({
            session_id: `session_${Date.now()}`,
            original_ai_id: shopper_id,
            shelf: shelf,
            total_dwell_time: dwell_time,
            last_seen_timestamp: currentTime
        });
    }

    // Send a 200 OK back to Python so it doesn't crash
    res.status(200).json({ status: "Success", action: wasMerged ? "Merged" : "Created" });
});

// --- DIAGNOSTIC ENDPOINT ---
// You can visit http://localhost:5001/api/data in your browser to see the live cleaned data
app.get('/api/data', (req, res) => {
    res.json(globalDatabase);
});

app.listen(PORT, () => {
    console.log(`\n===========================================`);
    console.log(`🚀 ShelfMate Backend running on Port ${PORT}`);
    console.log(`📡 Listening for Python AI Telemetry...`);
    console.log(`===========================================\n`);
});