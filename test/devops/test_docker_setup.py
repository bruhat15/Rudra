# test/devops/test_docker_setup.py
import pytest
import docker
import os

# This should be the image name and tag you intend to build
# For testing the build process itself, we use a temporary local tag.
IMAGE_NAME_FOR_BUILD_TEST = "rudra-ci-build-test"
TAG_FOR_BUILD_TEST = "latest" # or a unique tag like a commit sha
FULL_TEST_IMAGE_TAG = f"{IMAGE_NAME_FOR_BUILD_TEST}:{TAG_FOR_BUILD_TEST}"

@pytest.mark.docker # Remember to register this mark in pytest.ini or conftest.py
class TestDockerSetup:
    def test_dockerfile_exists(self):
        """Test if Dockerfile exists in the project root."""
        # Assuming pytest runs from the project root in CI
        assert os.path.exists('Dockerfile'), "Dockerfile not found in project root"

    def test_docker_image_builds_successfully(self, docker_environment): # Uses the client fixture
        """Test if the Docker image can be built successfully from the Dockerfile."""
        client = docker_environment
        print(f"Attempting to build Docker image with tag: {FULL_TEST_IMAGE_TAG} using Dockerfile from current directory ('.')")
        try:
            image, build_log_generator = client.images.build(
                path=".",  # Build context is the current directory (project root)
                tag=FULL_TEST_IMAGE_TAG,
                rm=True,    # Remove intermediate containers after build
                forcerm=True # Force removal even if build fails
            )
            # Consume the build log generator to ensure the build process completes
            build_log_output = []
            for chunk in build_log_generator:
                if 'stream' in chunk:
                    print(chunk['stream'].strip()) # Print Docker build output
                    build_log_output.append(chunk['stream'].strip())
                elif 'errorDetail' in chunk:
                    print(f"ERROR during build: {chunk['errorDetail']['message']}")
                    build_log_output.append(f"ERROR: {chunk['errorDetail']['message']}")
                    pytest.fail(f"Docker image build failed with error: {chunk['errorDetail']['message']}\nFull Log:\n{''.join(build_log_output)}")


            assert image is not None, "Docker image build process did not return an image object."
            assert FULL_TEST_IMAGE_TAG in image.tags, f"Image was built but not tagged as {FULL_TEST_IMAGE_TAG}"
            print(f"Docker image {FULL_TEST_IMAGE_TAG} built successfully.")

        except docker.errors.BuildError as e:
            # The build_log is often part of the exception object itself
            print("Docker build failed with BuildError:")
            log_output_from_exception = "".join([line.get('stream', str(line)) for line in e.build_log]) # Attempt to get stream or string representation
            print(log_output_from_exception)
            pytest.fail(f"Docker image build failed: {e}\nBuild Log:\n{log_output_from_exception}")
        except docker.errors.APIError as e:
            pytest.fail(f"Docker API error during build: {e}")
        finally:
            # Clean up the locally built test image
            try:
                if client.images.list(name=FULL_TEST_IMAGE_TAG):
                    client.images.remove(FULL_TEST_IMAGE_TAG, force=True)
                    print(f"Cleaned up test image: {FULL_TEST_IMAGE_TAG}")
            except docker.errors.ImageNotFound:
                print(f"Test image {FULL_TEST_IMAGE_TAG} not found for cleanup (already removed or build failed early).")
            except Exception as e_cleanup:
                print(f"Warning: Could not clean up test image {FULL_TEST_IMAGE_TAG}: {e_cleanup}")

    # The test_docker_hub_tag_exists is problematic for this workflow's timing.
    # It should be skipped or run in a different workflow after the image is pushed.
    @pytest.mark.skip(reason="This test requires the image to be pushed to Docker Hub first by another pipeline. Run separately or as an integration test post-push.")
    def test_docker_hub_image_can_be_pulled(self):
        """Test if the 'latest' image can be pulled from Docker Hub."""
        # Your Docker Hub username and the image name you actually push
        DOCKERHUB_USERNAME = "bruhat15"
        IMAGE_NAME_ON_HUB = "rudra-app" # Should match what your main CI/CD pushes
        image_to_pull = f"{DOCKERHUB_USERNAME}/{IMAGE_NAME_ON_HUB}:latest"

        print(f"Attempting to pull Docker image: {image_to_pull} from Docker Hub...")
        try:
            process = subprocess.run(
                ["docker", "pull", image_to_pull],
                capture_output=True,
                text=True,
                check=False
            )
            if process.returncode == 0:
                print(f"Successfully pulled {image_to_pull}.")
                client = docker.from_env()
                assert client.images.list(name=image_to_pull), f"Image {image_to_pull} was pulled but not found in local images."
            else:
                print(f"Failed to pull {image_to_pull}. STDOUT: {process.stdout} STDERR: {process.stderr}")
                pytest.fail(f"Failed to pull image {image_to_pull} from Docker Hub. Return code: {process.returncode}")
        except FileNotFoundError:
            pytest.fail("Docker CLI not found. Ensure Docker is installed and in PATH.")
        except Exception as e:
            pytest.fail(f"An error occurred while trying to pull the image: {e}")
        finally:
            try:
                client = docker.from_env()
                if client.images.list(name=image_to_pull):
                    client.images.remove(image_to_pull, force=True)
                    print(f"Cleaned up pulled image: {image_to_pull}")
            except Exception as e:
                print(f"Warning: Could not clean up pulled image {image_to_pull}: {e}")
