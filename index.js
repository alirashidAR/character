require('dotenv').config();
const axios = require('axios');

async function pingRenderBackend() {
  const RENDER_URL = process.env.RENDER_URL;

  if (!RENDER_URL) {
    console.error('Error: RENDER_URL environment variable is not set');
    process.exit(1);
  }

  console.log(`Pinging Render backend at ${RENDER_URL}...`);
  console.log(`Timestamp: ${new Date().toISOString()}`);

  const maxRetries = 3; 
  const retryDelay = 5000;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const startTime = Date.now();
      const response = await axios.get(RENDER_URL, {
        timeout: 30000, // 30-second timeout
      });
      const duration = Date.now() - startTime;
      console.log(`Ping successful! Status: ${response.status}`);
      console.log(`Response time: ${duration}ms`);
      return response;
    } catch (error) {
      console.error(`Ping attempt ${attempt} failed: ${error.message}`);
      if (attempt < maxRetries) {
        console.log(`Retrying in ${retryDelay / 1000} seconds...`);
        await new Promise((resolve) => setTimeout(resolve, retryDelay));
      } else {
        console.error('All retry attempts failed.');
        process.exit(1);
      }
    }
  }
}

// Execute ping and exit
(async () => {
  try {
    await pingRenderBackend();
    console.log('Ping completed successfully');
    process.exit(0);
  } catch (error) {
    console.error('Ping operation failed:', error.message);
    process.exit(1);
  }
})();