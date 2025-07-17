# Google Maps API Setup Guide

This guide will help you set up the required Google Maps APIs for the Smart US Travel Planner application.

## Prerequisites

1. A Google account
2. A Google Cloud project
3. Billing enabled on your Google Cloud project (required for API usage)

## Step-by-Step Setup

### 1. Create or Select a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Make sure billing is enabled for the project

### 2. Enable Required APIs

You need to enable the following APIs in your Google Cloud project:

#### Option A: Legacy APIs (Currently Deprecated)
- **Distance Matrix API**
- **Places API**
- **Static Maps API**

#### Option B: Newer APIs (Recommended)
- **Routes API** (replaces Distance Matrix API)
- **Places API (New)**
- **Static Maps API**

### 3. Enable APIs in Google Cloud Console

1. In the Google Cloud Console, go to **APIs & Services** > **Library**
2. Search for and enable each of the following APIs:

#### For Legacy APIs:
```
Distance Matrix API
Places API
Static Maps API
```

#### For Newer APIs:
```
Routes API
Places API (New)
Static Maps API
```

### 4. Create API Credentials

1. Go to **APIs & Services** > **Credentials**
2. Click **Create Credentials** > **API Key**
3. Copy the generated API key
4. (Optional) Restrict the API key to specific APIs for security

### 5. Configure API Key Restrictions (Recommended)

1. Click on your API key in the credentials list
2. Under **Application restrictions**, select **HTTP referrers** or **IP addresses**
3. Under **API restrictions**, select **Restrict key**
4. Select the APIs you enabled in step 3
5. Click **Save**

### 6. Add API Key to Your Application

1. Create a `.env` file in your project root (if it doesn't exist)
2. Add your API key:
```
GOOGLE_API_KEY=your_api_key_here
```

### 7. Test Your Setup

Run the API test script to verify everything is working:

```bash
python test_api_key.py
```

## API Usage and Billing

### Free Tier Limits

Google Maps APIs have generous free tiers:

- **Distance Matrix API**: 100,000 requests/month
- **Places API**: 28,500 requests/month
- **Static Maps API**: 100,000 requests/month
- **Routes API**: 100,000 requests/month

### Monitoring Usage

1. Go to **APIs & Services** > **Dashboard**
2. View usage statistics for each API
3. Set up billing alerts to avoid unexpected charges

## Troubleshooting

### Common Issues

#### 1. "REQUEST_DENIED" Error
**Cause**: API not enabled or API key doesn't have access
**Solution**: 
- Enable the required APIs in Google Cloud Console
- Check API key restrictions
- Verify billing is enabled

#### 2. "Legacy API" Error
**Cause**: Using deprecated APIs that are no longer enabled by default
**Solution**:
- Enable the legacy APIs specifically, or
- Upgrade to the newer APIs (Routes API, Places API New)

#### 3. "Billing Not Enabled" Error
**Cause**: Google Cloud project doesn't have billing enabled
**Solution**:
- Enable billing in Google Cloud Console
- Add a payment method

#### 4. "Quota Exceeded" Error
**Cause**: Exceeded free tier limits
**Solution**:
- Check usage in Google Cloud Console
- Wait for quota reset or upgrade billing plan

### API-Specific Issues

#### Distance Matrix API
- Ensure origins and destinations are valid addresses
- Check that travel mode is supported (driving, walking, bicycling, transit)

#### Places API
- Verify place_id is valid
- Check that requested fields are available for the place

#### Static Maps API
- Ensure map size is within limits (640x640 pixels max for free tier)
- Check that markers and paths are properly formatted

## Security Best Practices

1. **Restrict API Key**: Limit API key to specific APIs and domains
2. **Use Environment Variables**: Never hardcode API keys in source code
3. **Monitor Usage**: Set up alerts for unusual usage patterns
4. **Rotate Keys**: Regularly rotate API keys for security

## Cost Optimization

1. **Use Caching**: The application includes caching to reduce API calls
2. **Batch Requests**: Combine multiple requests when possible
3. **Monitor Usage**: Keep track of API usage to avoid unexpected charges
4. **Use Free Tier**: Stay within free tier limits for development

## Migration from Legacy to New APIs

If you're currently using legacy APIs and want to upgrade:

1. **Enable New APIs**: Enable Routes API and Places API (New)
2. **Update Code**: The application automatically tries newer APIs first
3. **Test Thoroughly**: Verify all functionality works with new APIs
4. **Monitor Performance**: New APIs may have different performance characteristics

## Support

For additional help:

1. **Google Cloud Documentation**: [Maps Platform Documentation](https://developers.google.com/maps/documentation)
2. **API Status**: [Google Cloud Status](https://status.cloud.google.com/)
3. **Community**: [Stack Overflow](https://stackoverflow.com/questions/tagged/google-maps-api)

## Example Configuration

Here's what your `.env` file should look like:

```env
# Google Maps API Configuration
GOOGLE_API_KEY=AIzaSyYourActualApiKeyHere

# Optional: Override default settings
CACHE_DURATION=3600
MAP_WIDTH=800
MAP_HEIGHT=600
MAP_ZOOM=12
```

## Testing Your Setup

After completing the setup, test your configuration:

```bash
# Test all APIs
python test_api_key.py

# Test the full application
python test_preferred_timing.py

# Start the application
python main.py
```

Visit `http://localhost:8000` to use the travel planner with real Google Maps data. 