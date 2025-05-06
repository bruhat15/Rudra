import pytest
import docker
import os
#testing pipeline
@pytest.mark.docker
class TestDockerSetup:
    def test_dockerfile_exists(self):
        """Test if Dockerfile exists"""
        assert os.path.exists('Dockerfile'), "Dockerfile not found"

    def test_docker_image_exists(self):
        """Test if Rudra image exists locally"""
        client = docker.from_env()
        images = client.images.list()
        image_tags = [tag for image in images for tag in image.tags]
        assert any('rudra:latest' in tag for tag in image_tags), "Rudra image not found"

    def test_docker_hub_tag_exists(self):
        """Test if Docker Hub tag exists"""
        client = docker.from_env()
        images = client.images.list()
        image_tags = [tag for image in images for tag in image.tags]
        assert any('adityamaller/rudra:latest' in tag for tag in image_tags), "Docker Hub tagged image not found"