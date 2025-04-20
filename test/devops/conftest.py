import pytest
import docker
import os
from time import sleep

@pytest.fixture(scope="session")
def docker_environment():
    """Setup and teardown for Docker tests"""
    client = docker.from_env()
    yield client
    
    # Cleanup: Stop any test containers
    for container in client.containers.list(filters={'name': 'rudra_test'}):
        container.stop()
        container.remove()

@pytest.fixture(scope="class")
def rudra_image(docker_environment):
    """Fixture to ensure Rudra image is available"""
    image_name = "rudra:latest"
    try:
        return docker_environment.images.get(image_name)
    except docker.errors.ImageNotFound:
        pytest.skip(f"Image {image_name} not found locally")

@pytest.fixture(scope="function")
def test_container(docker_environment, rudra_image):
    """Fixture to create and manage a test container"""
    container = docker_environment.containers.run(
        'rudra:latest',
        detach=True,
        ports={'8501/tcp': 8501},
        name=f'rudra_test_{pytest.current_test_name}',
        remove=True
    )
    
    # Wait for container to be ready
    sleep(2)
    
    yield container
    
    # Cleanup
    try:
        container.stop()
    except docker.errors.NotFound:
        pass  # Container already removed

@pytest.fixture(scope="session")
def project_root():
    """Fixture to provide project root directory"""
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))