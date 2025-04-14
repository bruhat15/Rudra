#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t rudra .

# Run tests in Docker
echo "Running tests in Docker..."
docker run rudra

# Check if tests passed
if [ $? -eq 0 ]; then
  echo "✅ All tests passed!"
else
  echo "❌ Some tests failed."
  exit 1
fi 