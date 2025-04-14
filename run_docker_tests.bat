@echo off
echo Building Docker image...
docker build -t rudra .

echo Running tests in Docker...
docker run rudra

if %ERRORLEVEL% EQU 0 (
  echo ✅ All tests passed!
) else (
  echo ❌ Some tests failed.
  exit /b 1
) 