FROM secretflow/ubuntu-base-ci:latest

# Create a non-root user 'vscode' with sudo privileges
# The user ID 1000 is common for the primary user in Linux distributions.
# The GID 1000 corresponds to the user's primary group.
RUN groupadd --gid 1000 vscode && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set the default user for subsequent commands
USER vscode

# Set the working directory inside the container
WORKDIR /workspaces/spu

# Set default shell to bash for the vscode user
ENV SHELL /bin/bash

# You can add more SPU-specific dependencies here if needed,
# although 'make install-dev-deps' in postCreateCommand should handle most project dependencies.

# Keep the container running (optional, useful for debugging setup)
# CMD ["sleep", "infinity"]
