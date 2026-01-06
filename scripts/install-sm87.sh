#!/bin/bash
# =============================================================================
# TEI Installation Script for SM87 Devices (NVIDIA Jetson Orin)
# =============================================================================
# Supported devices: Jetson AGX Orin, Jetson Orin NX, Jetson Orin Nano
# CUDA Compute Capability: 8.7
# =============================================================================

set -e

# Configuration
REPO="thomas-hiddenpeak/RMinte-Orin-TEI"
RELEASE_TAG="v0.6.0-qwen3-reranker-orin"
BINARY_NAME="text-embeddings-router"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.cargo/bin}"
TMP_DIR="/tmp/tei-install"
FORCE_INSTALL="${FORCE_INSTALL:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  TEI Installation Script for SM87 (Jetson Orin)              ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running on ARM64
check_architecture() {
    local arch=$(uname -m)
    if [[ "$arch" != "aarch64" ]]; then
        print_error "This binary is built for aarch64 (ARM64), but you're running on $arch"
        exit 1
    fi
    print_step "Architecture check passed: $arch"
}

# Check CUDA availability
check_cuda() {
    if ! command -v nvidia-smi &> /dev/null; then
        print_warn "nvidia-smi not found. CUDA may not be properly installed."
        print_warn "TEI requires CUDA 12.x for optimal performance."
    else
        local cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        print_step "NVIDIA driver detected: $cuda_version"
        
        # Check compute capability
        local cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
        if [[ "$cc" == "8.7" ]]; then
            print_step "Compute capability: $cc (SM87 - Perfect match!)"
        else
            print_warn "Compute capability: $cc (This binary is optimized for SM87)"
        fi
    fi
}

# Check for required libraries
check_dependencies() {
    print_info "Checking dependencies..."
    
    local missing_deps=()
    
    # Check for libcuda
    if ! ldconfig -p | grep -q "libcuda.so"; then
        missing_deps+=("libcuda")
    fi
    
    # Check for libcublas
    if ! ldconfig -p | grep -q "libcublas.so"; then
        missing_deps+=("libcublas")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_warn "Missing libraries: ${missing_deps[*]}"
        print_warn "Please ensure CUDA toolkit is properly installed"
    else
        print_step "All required libraries found"
    fi
}

# Check for existing installation
check_existing_installation() {
    local target_path="$INSTALL_DIR/$BINARY_NAME"
    
    if [[ -f "$target_path" ]]; then
        print_warn "Existing installation found: $target_path"
        
        # Try to get version info
        local existing_version=""
        if "$target_path" --version &> /dev/null; then
            existing_version=$("$target_path" --version 2>&1 | head -1)
            print_info "Existing version: $existing_version"
        fi
        
        if [[ "$FORCE_INSTALL" == "true" ]]; then
            print_info "Force install enabled, will overwrite existing installation"
            return 0
        fi
        
        # Interactive prompt (only if stdin is a terminal)
        if [[ -t 0 ]]; then
            echo ""
            echo -e "${YELLOW}An existing installation was found.${NC}"
            echo -e "  Current: $target_path"
            echo -e "  New version: $RELEASE_TAG"
            echo ""
            read -p "Do you want to overwrite it? [y/N] " -n 1 -r
            echo ""
            
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_info "Installation cancelled by user"
                exit 0
            fi
            
            # Backup existing binary
            local backup_path="${target_path}.backup.$(date +%Y%m%d%H%M%S)"
            print_info "Creating backup: $backup_path"
            cp "$target_path" "$backup_path"
            print_step "Backup created"
        else
            # Non-interactive mode without FORCE_INSTALL
            print_error "Existing installation found. Use FORCE_INSTALL=true to overwrite"
            print_info "Example: FORCE_INSTALL=true $0"
            exit 1
        fi
    fi
}

# Ensure install directory exists
ensure_install_dir() {
    if [[ ! -d "$INSTALL_DIR" ]]; then
        print_info "Creating install directory: $INSTALL_DIR"
        mkdir -p "$INSTALL_DIR"
        print_step "Directory created"
    fi
    
    # Check if directory is in PATH
    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
        print_warn "$INSTALL_DIR is not in your PATH"
        print_info "Add this to your shell profile (~/.bashrc or ~/.zshrc):"
        echo -e "    ${YELLOW}export PATH=\"\$HOME/.cargo/bin:\$PATH\"${NC}"
    fi
}

# Download the binary
download_binary() {
    print_info "Downloading TEI binary from GitHub..."
    
    mkdir -p "$TMP_DIR"
    cd "$TMP_DIR"
    
    local download_url="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${BINARY_NAME}"
    
    if command -v curl &> /dev/null; then
        curl -L -o "$BINARY_NAME" "$download_url" --progress-bar
    elif command -v wget &> /dev/null; then
        wget -O "$BINARY_NAME" "$download_url" --show-progress
    else
        print_error "Neither curl nor wget found. Please install one of them."
        exit 1
    fi
    
    if [[ ! -f "$BINARY_NAME" ]]; then
        print_error "Failed to download binary"
        exit 1
    fi
    
    local file_size=$(stat -c%s "$BINARY_NAME" 2>/dev/null || stat -f%z "$BINARY_NAME" 2>/dev/null)
    print_step "Downloaded: $BINARY_NAME ($(numfmt --to=iec-i --suffix=B $file_size 2>/dev/null || echo "${file_size} bytes"))"
}

# Install the binary
install_binary() {
    print_info "Installing to $INSTALL_DIR..."
    
    # Check if we need sudo
    if [[ -w "$INSTALL_DIR" ]]; then
        cp "$TMP_DIR/$BINARY_NAME" "$INSTALL_DIR/"
        chmod +x "$INSTALL_DIR/$BINARY_NAME"
    else
        print_info "Requesting sudo access for installation..."
        sudo cp "$TMP_DIR/$BINARY_NAME" "$INSTALL_DIR/"
        sudo chmod +x "$INSTALL_DIR/$BINARY_NAME"
    fi
    
    print_step "Installed: $INSTALL_DIR/$BINARY_NAME"
}

# Verify installation
verify_installation() {
    print_info "Verifying installation..."
    
    if [[ -x "$INSTALL_DIR/$BINARY_NAME" ]]; then
        print_step "Binary is executable"
        
        # Try to run with --help
        if "$INSTALL_DIR/$BINARY_NAME" --help &> /dev/null; then
            print_step "Binary runs successfully"
        else
            print_warn "Binary may have missing dependencies"
        fi
    else
        print_error "Installation verification failed"
        exit 1
    fi
}

# Cleanup
cleanup() {
    print_info "Cleaning up..."
    rm -rf "$TMP_DIR"
    print_step "Cleanup complete"
}

# Print usage instructions
print_usage() {
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Installation Complete!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Quick Start:"
    echo ""
    echo "  1. Start the server with a model:"
    echo -e "     ${YELLOW}text-embeddings-router --model-id BAAI/bge-small-en-v1.5${NC}"
    echo ""
    echo "  2. For Qwen3-Reranker (requires fp16 model):"
    echo -e "     ${YELLOW}text-embeddings-router --model-id /path/to/qwen3-reranker-fp16${NC}"
    echo ""
    echo "  3. Test with curl:"
    echo -e "     ${YELLOW}curl http://localhost:3000/embed \\
       -H 'Content-Type: application/json' \\
       -d '{\"inputs\": \"Hello world\"}'${NC}"
    echo ""
    echo "Documentation:"
    echo "  https://github.com/${REPO}/blob/feature/qwen3-reranker/docs/QWEN3_RERANKER.md"
    echo ""
}

# Main execution
main() {
    print_header
    
    echo "Release: $RELEASE_TAG"
    echo "Target:  $INSTALL_DIR/$BINARY_NAME"
    echo ""
    
    check_architecture
    check_cuda
    check_dependencies
    ensure_install_dir
    check_existing_installation
    download_binary
    install_binary
    verify_installation
    cleanup
    print_usage
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "TEI Installation Script for SM87 Devices (Jetson Orin)"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --version, -v  Show version information"
        echo "  --force, -f    Force install, overwrite existing installation"
        echo ""
        echo "Environment Variables:"
        echo "  INSTALL_DIR      Installation directory (default: ~/.cargo/bin)"
        echo "  FORCE_INSTALL    Set to 'true' to skip overwrite confirmation"
        echo ""
        echo "Examples:"
        echo "  $0                              # Install to ~/.cargo/bin"
        echo "  $0 --force                      # Force overwrite existing"
        echo "  INSTALL_DIR=/usr/local/bin $0   # Install to /usr/local/bin"
        echo "  FORCE_INSTALL=true $0           # Non-interactive force install"
        exit 0
        ;;
    --version|-v)
        echo "TEI Installation Script"
        echo "Release: $RELEASE_TAG"
        echo "Repository: $REPO"
        exit 0
        ;;
    --force|-f)
        FORCE_INSTALL=true
        main
        ;;
    *)
        main
        ;;
esac
