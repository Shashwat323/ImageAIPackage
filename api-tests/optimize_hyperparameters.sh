# Define the API URL
API_URL="http://127.0.0.1:5000/hyperparameter-tuning"

# Define the JSON configuration file path
JSON_CONFIG_FILE="hyperparameter_tuning_config.json"

# Check if the configuration file exists
if [[ ! -f $JSON_CONFIG_FILE ]]; then
  echo "Error: Configuration file '$JSON_CONFIG_FILE' not found!"
  exit 1
fi

# Send a POST request with the JSON file
echo "Sending hyperparameter tuning request to $API_URL..."
RESPONSE=$(curl -X POST -H "Content-Type: application/json" -d @"$JSON_CONFIG_FILE" $API_URL)

# Print the response
echo "Response:"
echo $RESPONSE
read -n 1 -s